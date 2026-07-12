import string

"""
//////////////////
/// TOKENIZERS ///
//////////////////

Text tokenizers for preprocessing raw text into integer token ids.
Each tokenizer builds a vocabulary during fit() and reuses that vocabulary
to transform new text into model-ready token sequences.
"""


class NotFittedError(RuntimeError):
    """
    Raised when a tokenizer is used to transform text before fit() has built
    its vocabulary.
    """


class Tokenizer:
    """
    Base tokenizer class that stores vocabulary mappings and common text
    normalization options. Specific tokenizers inherit from this class and
    implement their own fit and transform logic.
    """
    def __init__(
        self,
        lowercase: bool = True,
        strip_punctuation: bool = True,
        normalize_whitespace: bool = True,
    ) -> None:
        """
        Initializes tokenizer vocabulary, special tokens, and normalization
        flags used before fitting or transforming text.
        """
        self.token_to_id = {}
        self.id_to_token = {}
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.fitted = False
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        # Build the punctuation translation table once instead of on every
        # normalize_text call, since it never changes after construction
        self._punctuation_table = str.maketrans(
            {char: " " for char in string.punctuation}
        )

    def normalize_text(self, text: str) -> str:
        """
        Applies configured text normalization steps before tokenization.
        Lowercases text, replaces punctuation with spaces, and collapses
        repeated whitespace when the corresponding flags are enabled.
        """
        # Normalize casing so vocabulary entries are consistent
        if self.lowercase:
            text = text.lower()

        # Replace punctuation with spaces to keep word boundaries clean
        if self.strip_punctuation:
            text = text.translate(self._punctuation_table)

        # Collapse tabs, newlines, and repeated spaces into single spaces
        if self.normalize_whitespace:
            text = " ".join(text.split())

        return text

    def fit(self, corpus: str):
        """
        Builds the tokenizer vocabulary from a corpus. Implemented by each
        specific tokenizer subclass.
        """
        raise NotImplementedError

    def transform(self, text: str):
        """
        Converts text into token ids using the fitted vocabulary. Implemented
        by each specific tokenizer subclass.
        """
        raise NotImplementedError

    def fit_transform(self, corpus: str):
        """
        Convenience method that fits the tokenizer and transforms the corpus
        in one step, mirroring the scaler API.
        """
        self.fit(corpus)
        return self.transform(corpus)

    def tokenize(self, text: str):
        """
        Alias for transform to match common tokenizer APIs.
        """
        return self.transform(text)

    def decode(self, token_ids: list[int]):
        """
        Converts a sequence of token ids back into their string tokens using
        the fitted vocabulary. Unknown ids are mapped to the <unk> token.
        """
        self._ensure_fitted()
        return [self.id_to_token.get(token_id, self.unk_token) for token_id in token_ids]

    def _ensure_fitted(self):
        """
        Validates that a tokenizer has already built its vocabulary.
        """
        if not self.fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} instance has not yet been fitted"
            )


class CharacterTokenizer(Tokenizer):
    """
    Tokenizer that treats every normalized character as a token.
    Useful for small vocabularies and character-level language modeling.
    """
    def __init__(self) -> None:
        """
        Initializes a character tokenizer with the default normalization rules.
        """
        super().__init__()

    def fit(self, corpus: str):
        """
        Builds a character vocabulary from the unique characters in a corpus.
        Special padding and unknown tokens are reserved at ids 0 and 1.
        """
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        corpus = self.normalize_text(corpus)
        character_corpus = sorted(set(corpus))

        # Add each observed character to the vocabulary in deterministic order
        for character in character_corpus:
            self.token_to_id[character] = len(self.token_to_id)

        self.id_to_token = {id: token for token, id in self.token_to_id.items()}
        self.fitted = True

    def transform(self, text: str):
        """
        Converts text into character token ids using the fitted vocabulary.
        Unknown characters are mapped to the <unk> token id.
        """
        self._ensure_fitted()

        output_ids = []
        input_tokens = list(self.normalize_text(text))
        unknown_token_id = self.token_to_id[self.unk_token]

        for tok in input_tokens:
            output_ids.append(self.token_to_id.get(tok, unknown_token_id))

        return output_ids


class WordTokenizer(Tokenizer):
    """
    Tokenizer that treats each whitespace-separated word as a token.
    Builds compact word-level vocabularies for simple NLP workflows.
    """
    def __init__(self) -> None:
        """
        Initializes a word tokenizer with the default normalization rules.
        """
        super().__init__()

    def fit(self, corpus: str):
        """
        Builds a word vocabulary from the unique normalized words in a corpus.
        Special padding and unknown tokens are reserved at ids 0 and 1.
        """
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        corpus = self.normalize_text(corpus)
        word_corpus = sorted(set(corpus.split()))

        # Add each observed word to the vocabulary in deterministic order
        for word in word_corpus:
            self.token_to_id[word] = len(self.token_to_id)

        self.id_to_token = {id: token for token, id in self.token_to_id.items()}
        self.fitted = True

    def transform(self, text: str):
        """
        Converts text into word token ids using the fitted vocabulary.
        Unknown words are mapped to the <unk> token id.
        """
        self._ensure_fitted()

        output_ids = []
        input_tokens = self.normalize_text(text).split()
        unknown_token_id = self.token_to_id[self.unk_token]

        for tok in input_tokens:
            output_ids.append(self.token_to_id.get(tok, unknown_token_id))

        return output_ids


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer.
    Pre-tokenizes the corpus into words, represents each word as its raw UTF-8
    bytes, and repeatedly merges the most frequent adjacent pair within words
    to build a compact subword vocabulary. Merges never cross word boundaries,
    matching how production byte-level BPE pre-tokenizes.
    """
    def __init__(
        self,
        num_merges: int = 50,
        min_frequency: int = 1,
        encoding: str = "latin-1",
    ) -> None:
        """
        Initializes BPE training parameters and empty merge tables.

        num_merges: Maximum number of pair merges to learn
        min_frequency: Minimum pair frequency required to create a merge
        encoding: Byte encoding used for the initial vocabulary
        """
        super().__init__()
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.encoding = encoding
        self.byte_tokenization_size = 256
        self.merges = []
        self.merge_ranks = {}
        self.merge_to_id = {}

    def _create_byte_vocabulary(self):
        """
        Creates the initial byte-level vocabulary.
        Each of the 256 byte values maps to a single display character. latin-1
        is used because it is the only encoding with a 1:1 mapping for every
        byte, giving each id a readable token string.
        """
        return {
            i: bytes([i]).decode(self.encoding)
            for i in range(self.byte_tokenization_size)
        }

    def _byte_tokenize(self, text: str):
        """
        Converts text into its raw UTF-8 byte ids.
        UTF-8 keeps this a true byte-level tokenizer: every byte is in 0-255,
        so any Unicode input is representable instead of only latin-1 text.
        """
        return list(text.encode("utf-8"))

    def _create_byte_splits(self, words: list[str]):
        """
        Counts words and represents each unique word as its list of UTF-8 byte
        ids. These per-word byte sequences are the starting point for merging.
        """
        word_counts = {}
        splits = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            splits[word] = self._byte_tokenize(word)

        return word_counts, splits

    def _count_pairs(self, word_counts, splits):
        """
        Counts adjacent token pairs across all word splits, weighted by word
        frequency. Only pairs within a word are counted, so no pair ever spans
        a word boundary. These counts determine which pair is merged next.
        """
        pair_counts = {}
        for word, split_tokens in splits.items():
            count = word_counts[word]
            for pair in zip(split_tokens, split_tokens[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        return pair_counts

    def _merge_pair(self, tokens, pair, new_id):
        """
        Replaces every non-overlapping occurrence of a token pair with a new id.
        """
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _merge_pair_all_splits(self, splits, pair, new_id):
        """
        Replaces every non-overlapping occurrence of a pair with new_id in
        every word split.
        """
        return {
            word: self._merge_pair(split_tokens, pair, new_id)
            for word, split_tokens in splits.items()
        }

    def fit(self, corpus: str):
        """
        Learns BPE merges from the corpus and builds token/id lookup tables.
        The corpus is pre-tokenized into words, each word starts as its raw
        UTF-8 bytes, and one merged token is added per iteration by merging the
        most frequent adjacent pair within words.
        """
        self.id_to_token = self._create_byte_vocabulary()
        self.merges = []
        self.merge_ranks = {}
        self.merge_to_id = {}

        words = self.normalize_text(corpus).split()
        word_counts, splits = self._create_byte_splits(words)

        # Repeatedly merge the most frequent adjacent pair within words
        for _ in range(self.num_merges):
            pair_counts = self._count_pairs(word_counts, splits)
            if not pair_counts:
                break

            most_frequent_pair = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[most_frequent_pair] < self.min_frequency:
                break

            tok1, tok2 = most_frequent_pair
            new_id = len(self.id_to_token)
            splits = self._merge_pair_all_splits(splits, most_frequent_pair, new_id)
            self.merges.append(most_frequent_pair)
            self.merge_ranks[most_frequent_pair] = len(self.merge_ranks)
            self.merge_to_id[most_frequent_pair] = new_id
            self.id_to_token[new_id] = self.id_to_token[tok1] + self.id_to_token[tok2]

        # BPE deliberately does not build a token_to_id map: two different
        # merges can produce the same concatenated string (e.g. ('a','bc') and
        # ('ab','c') both yield "abc"), which would collapse distinct ids.
        # Encoding uses merge_to_id and decoding uses id_to_token instead.
        self.fitted = True

    def transform(self, text: str):
        """
        Tokenizes text by pre-splitting into words and applying learned BPE
        merges within each word, then concatenating the per-word ids. Merges
        are applied in the order they were learned. Word boundaries (spaces)
        are dropped, so the result reconstructs the normalized text without
        its spaces.
        """
        self._ensure_fitted()

        output_ids = []
        for word in self.normalize_text(text).split():
            word_tokens = self._byte_tokenize(word)

            # Apply learned merges in the same order they were created
            for pair in self.merges:
                word_tokens = self._merge_pair(
                    word_tokens,
                    pair,
                    self.merge_to_id[pair],
                )

            output_ids.extend(word_tokens)

        return output_ids


class WordPiece(Tokenizer):
    """
    WordPiece tokenizer that learns subword units from a text corpus.
    Starts with character-level word pieces, then repeatedly merges adjacent
    pieces using the WordPiece scoring rule.
    """
    def __init__(
        self,
        iterations: int = 100,
        continuation_prefix: str = "##",
    ) -> None:
        """
        Initializes a WordPiece tokenizer with the default normalization rules.

        iterations: Maximum number of merge steps to learn during fit
        continuation_prefix: Marks pieces that continue an existing word
        """
        super().__init__()
        self.iterations = iterations
        self.continuation_prefix = continuation_prefix

    def fit(self, corpus: str):
        """
        Learns a WordPiece vocabulary from the corpus.
        Training starts with character pieces, then adds the highest-scoring
        merged pair at each iteration.
        """
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        words_corpus = self.normalize_text(corpus).split()
        word_counts, splits = self.create_splits_and_word_counts(words_corpus)

        # Seed the vocabulary with every observed character piece
        for split_tokens in splits.values():
            for token in split_tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.token_to_id)

        # Repeatedly merge the highest-scoring adjacent word pieces
        for _ in range(self.iterations):
            pair_scores = self.compute_scores(word_counts, splits)
            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.__getitem__)
            splits = self.merge_pair_all_splits(splits, best_pair)
            merged_token = self.pair_to_token(best_pair)

            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.token_to_id)

        self.id_to_token = {id: token for token, id in self.token_to_id.items()}
        self.fitted = True

    def transform(self, text: str):
        """
        Converts text into WordPiece token ids using the fitted vocabulary.
        Each word is greedily segmented by the longest matching pieces.
        """
        self._ensure_fitted()

        output_ids = []
        unknown_token_id = self.token_to_id[self.unk_token]

        for word in self.normalize_text(text).split():
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                output_ids.append(self.token_to_id.get(token, unknown_token_id))

        return output_ids

    def _tokenize_word(self, word: str):
        """
        Greedily tokenizes one word using the longest matching word pieces.
        Returns <unk> when any part of the word cannot be matched.
        """
        start = 0
        word_tokens = []

        while start < len(word):
            end = len(word)
            matched_token = None

            while start < end:
                candidate = self._format_piece(word[start:end], start)

                if candidate in self.token_to_id:
                    matched_token = candidate
                    break

                end -= 1

            if matched_token is None:
                return [self.unk_token]

            word_tokens.append(matched_token)
            start = end

        return word_tokens

    def _format_piece(self, piece: str, start: int):
        """
        Adds the continuation prefix to pieces that do not start a word.
        """
        if start == 0:
            return piece
        return f"{self.continuation_prefix}{piece}"

    def merge_pair_all_splits(self, splits, pair):
        """
        Replaces every non-overlapping occurrence of a pair in all word splits.
        """
        for word, split_tokens in splits.items():
            new_split = []
            i = 0

            while i < len(split_tokens):
                if (
                    i < len(split_tokens) - 1
                    and (split_tokens[i], split_tokens[i + 1]) == pair
                ):
                    new_split.append(self.pair_to_token(pair))
                    i += 2
                else:
                    new_split.append(split_tokens[i])
                    i += 1

            splits[word] = new_split

        return splits

    def pair_to_token(self, pair):
        """
        Converts a merge pair into the token that represents the merged piece.
        """
        return pair[0] + pair[1].removeprefix(self.continuation_prefix)

    def create_splits_and_word_counts(self, words_corpus):
        """
        Counts normalized words and creates the initial character splits.
        First characters are plain tokens; later characters are continuation
        pieces prefixed with the configured continuation marker.
        """
        word_counts = {}
        splits = {}

        for word in words_corpus:
            word_counts[word] = word_counts.get(word, 0) + 1
            splits[word] = [word[0]]

            for character in word[1:]:
                splits[word].append(f"{self.continuation_prefix}{character}")

        return word_counts, splits

    def compute_scores(self, word_counts, splits):
        """
        Computes WordPiece scores for every adjacent pair.
        Score = pair frequency divided by the product of both token frequencies.
        """
        token_freqs = {}
        pair_freqs = {}
        pair_scores = {}

        for word, count in word_counts.items():
            split_tokens = splits[word]

            for token in set(split_tokens):
                token_freqs[token] = (
                    token_freqs.get(token, 0) + split_tokens.count(token) * count
                )

            for i in range(len(split_tokens) - 1):
                pair = (split_tokens[i], split_tokens[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + count

        for pair, pair_frequency in pair_freqs.items():
            pair_scores[pair] = pair_frequency / (
                token_freqs[pair[0]] * token_freqs[pair[1]]
            )

        return pair_scores
