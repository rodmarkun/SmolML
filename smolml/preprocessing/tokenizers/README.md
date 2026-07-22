# SmolML - Tokenizers: Breaking Languages into Numbers

As you may already know, models cannot understand language (words, characters, etc) like a human would do. They only understand numbers. This is why **tokenizers** exist! They convert language into a vector of numbers, letting a model talk with us. The basic intuition is that we take a *unit of language* and convert it into a number.

What is a unit of language? Well, that's where the main tokenizers differ, and where the actual science of tokenization resides. We can do something super simple, like converting each word we see into a number, or we can do way more elaborate stuff like BPE, as we'll see.

For example, here the word "hello" corresponds to the number 1, the word "goodbye" to the number 2, and so on:

```python
{"hello": 1, "goodbye": 2, ...}
```

As you can see, this means a tokenizer can be as simple as a lookup table from language to numbers. All the language units we register in this table conform our **vocabulary**.

## `Tokenizer` 

First off, we'll create a base class `Tokenizer` that will serve as a foundation for all of our tokenizers, holding all of the shared behavior:

```python
class Tokenizer:
    def __init__(self, lowercase=True, strip_punctuation=True, normalize_whitespace=True) -> None:
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
```

The two dictionaries `token_to_id` and `id_to_token` are exactly that lookup table we talked about (and its reverse, so we can also go from numbers back to language!).

One of the aspects we want this class to handle is **text normalization**. Text is messy, and we may (or may not) want to reduce the amount of noise in our input. This, however, can cause us to lose some information as well. See the following examples:

* "I may eat a burrito"
* "I ate a burrito in May"

In this case, "May" and "may" are completely different words. If we normalize our text into all lowercase, we lose this distinction. However, think that storing both the lowercase and capitalized version of each word would double the size of our vocabulary!

As we just care about teaching the fundamentals of tokenizers, we won't complicate ourselves and will work on a lowercased, stripped-punctuation and normalized-whitespace input. But feel free to modify this however you want and see how everything changes!

```python
def normalize_text(self, text: str) -> str:
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
```

You may have also noticed we defined two **special tokens** in the constructor: the padding token `<pad>` and the unknown token `<unk>`.

* The **padding token** exists because models usually process text in batches, and every sequence in a batch must have the same length. Real sentences rarely have the exact same number of tokens, so we fill the shorter ones with `<pad>` until they match the longest one. It's a token that deliberately means "nothing here, ignore me".
* The **unknown token** is our safety net. What happens if, after fitting, we ask the tokenizer to transform a word (or character) it has never seen before? It's not in the vocabulary, so there's no ID for it! Instead of crashing, we map anything we don't recognize to `<unk>`. We lose the original information, sure, but the show can go on.

There are more of these special tokens some tokenizers use, but we won't go into too much detail for simplicity. Just be aware that they exist and can be used to fulfill different purposes.

Finally, the API. If you read the [Scalers](https://github.com/rodmarkun/SmolML/tree/main/smolml/preprocessing/scalers) section, this will look very familiar:

* `fit(corpus)` shows the tokenizer a corpus of text and lets it extract all the tokens it sees, based on its work methodology.
* `transform(text)` takes an input string and converts it to token ids, using what we learnt during `fit()`.
* `fit_transform(corpus)` does both in one step, for pure convenience.
* `decode(token_ids)` goes the other way: from token ids back to their string tokens, using `id_to_token`.

Each specific tokenizer implements its own `fit()` and `transform()`. Let's build some!

## `CharacterTokenizer`

This one's easy! We take the characters of a string as our language unit, and assign each one a number:

<div align="center">
  <img src="../../../images/tokenizers/CharacterTokenizer.gif" alt="CharacterTokenizer splitting text into characters and mapping them to token ids" width="850">
</div>

```python
def fit(self, corpus: str):
    self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
    corpus = self.normalize_text(corpus)
    character_corpus = sorted(set(corpus))

    # Add each observed character to the vocabulary in deterministic order
    for character in character_corpus:
        self.token_to_id[character] = len(self.token_to_id)

    self.id_to_token = {id: token for token, id in self.token_to_id.items()}
    self.fitted = True
```

We reserve ids 0 and 1 for our special tokens, then give every unique character in the corpus its own id. Transforming is just looking up each character (falling back to `<unk>` for characters we've never seen):

```python
def transform(self, text: str):
    self._ensure_fitted()

    output_ids = []
    input_tokens = list(self.normalize_text(text))
    unknown_token_id = self.token_to_id[self.unk_token]

    for tok in input_tokens:
        output_ids.append(self.token_to_id.get(tok, unknown_token_id))

    return output_ids
```

This gives us a *tiny* vocabulary (there are only so many characters!), but at a cost: every single character becomes one token, so our sequences get really long.

## `WordTokenizer`

Another easy one. Same deal, but instead of characters we use whole words as language units:

<div align="center">
  <img src="../../../images/tokenizers/WordTokenizer.gif" alt="WordTokenizer splitting text into words and mapping an unseen word to the unknown token" width="850">
</div>

```python
def fit(self, corpus: str):
    self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
    corpus = self.normalize_text(corpus)
    word_corpus = sorted(set(corpus.split()))

    # Add each observed word to the vocabulary in deterministic order
    for word in word_corpus:
        self.token_to_id[word] = len(self.token_to_id)

    self.id_to_token = {id: token for token, id in self.token_to_id.items()}
    self.fitted = True
```

The `transform()` method is the same as the character one, just splitting the text into words instead of characters.

This trades the character tokenizer's problem for the opposite one: sequences become much shorter (one token per word!), but the vocabulary explodes, since we need an entry for *every unique word* in the corpus. And any word we didn't see during `fit()` becomes `<unk>`. Saw "eat" and "eating" but not "eats"? Too bad, `<unk>` it is, even though the three words are obviously related.

There has to be a middle ground between characters and words, right? Right! That's exactly what subword tokenizers like BPE are about.

## `BPETokenizer`

Now we start with more interesting algorithms. BPE stands for **Byte Pair Encoding**.

Before diving in, some context: BPE comes in two main flavors, and they only differ in what the *starting symbols* are. The merging algorithm itself is exactly the same in both!

* **Character-level BPE** (the original formulation for NLP, from around 2016) starts from the characters of the corpus. It's very readable ("e" + "s" merges into "es"), but it has a weakness: if the tokenizer later encounters a character that wasn't in the training corpus (an emoji, an accented letter, a Chinese character...), it has no choice but to fall back to `<unk>`.
* **Byte-level BPE** (popularized by GPT-2, and what basically every modern LLM uses today) starts from the raw bytes of the text instead. Any text, in any language, is ultimately just a sequence of bytes, and a byte can only take 256 possible values. This means a base vocabulary of just 256 entries covers *everything*. No `<unk>` needed, ever!

We chose byte-level here for two reasons: it completely removes the unknown-token problem, and it's the variant you'll actually find inside real LLMs like GPT or Llama. Plus, for plain English text, bytes and characters happen to be the same thing anyway, so we don't even lose readability in our examples.

Now, just using bytes alone would have a similar effect as the `CharacterTokenizer`: a super long sequence for each text, with a very small vocab size. This is where the concept of byte *pairing* enters:

We look at which pairs of adjacent tokens repeat the most across the whole corpus, and iteratively **merge** the most common one into a new piece of our vocabulary! Frequent fragments like "th" or "ing" quickly become single tokens. This makes the sequence length much smaller while also keeping our vocabulary relatively small. The middle ground we were looking for!

<div align="center">
  <img src="../../../images/tokenizers/BPETokenizer.gif" alt="Byte Pair Encoding counting adjacent byte pairs and merging the most frequent pair" width="850">
</div>

One important detail: instead of taking byte pairs as-is across the whole text, we only consider byte pairs *within the same word*. Merges are never allowed to cross a word boundary. Why? If we let merges span across words, the most frequent pairs would quickly become things like the end of one word glued to the start of the next ("e" + "t" from "the tokenizer"), and we'd start creating tokens that mix pieces of two different words. Those tokens don't capture the internal structure of words, they capture which words happen to appear together in our training corpus, and that generalizes terribly to new text. Keeping merges inside word boundaries forces the algorithm to learn actual subword units: prefixes, suffixes, roots... This is also what production tokenizers do (GPT-2, for example, pre-splits text with a regex before doing any merging). As a bonus, it makes training way cheaper: we can count each unique word once and just weight its pairs by how many times the word appears, instead of scanning the entire corpus on every merge.

### Training BPE

Step 1: we create the initial vocabulary, which is simply all 256 possible byte values, where each byte is its own token ID (0 to 255). To turn actual text into bytes we encode it with UTF-8, which can represent any Unicode text as a sequence of bytes in that 0-255 range:

```python
def _create_byte_vocabulary(self):
    return {
        i: bytes([i]).decode(self.encoding)
        for i in range(self.byte_tokenization_size)
    }

def _byte_tokenize(self, text: str):
    return list(text.encode("utf-8"))
```

You may wonder what that `self.encoding` (latin-1 by default) is doing there. We use it just to *display* each byte ID as a readable character, since latin-1 is the only encoding that maps every one of the 256 byte values to exactly one character. The actual tokenization is pure UTF-8 bytes.

Step 2: we count all byte pairs across all words. Note how we count each unique word once and weight its pairs by the word's frequency, like we promised:

```python
def _count_pairs(self, word_counts, splits):
    pair_counts = {}
    for word, split_tokens in splits.items():
        count = word_counts[word]
        for pair in zip(split_tokens, split_tokens[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts
```

Step 3: we get the most frequent pair, assign it a new ID, and go through all word splits substituting every occurrence of the pair with the newly formed token (which is the combination of both!):

```python
def _merge_pair(self, tokens, pair, new_id):
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
```

Step 4: keep going until we reach our max number of merges, or until no pair appears at least `min_frequency` times (these are parameters we can adjust as we want). Putting it all together in `fit()`:

```python
def fit(self, corpus: str):
    self.id_to_token = self._create_byte_vocabulary()
    self.merges = []
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
        self.merge_to_id[most_frequent_pair] = new_id
        self.id_to_token[new_id] = self.id_to_token[tok1] + self.id_to_token[tok2]

    self.fitted = True
```

A curious detail: unlike our other tokenizers, BPE deliberately does *not* build a `token_to_id` map. Two different merges can produce the same concatenated string (for example, ("a", "bc") and ("ab", "c") both yield "abc"), which would collapse two distinct ids into one dictionary key! Encoding uses `merge_to_id` and decoding uses `id_to_token` instead.

### Transforming with BPE

Then, when we transform an input text with our tokenizer:

1. For each word, we convert it into its raw UTF-8 bytes.
2. We replay every merge learnt during `fit()`, in the same order they were created.
3. Concatenate the ids of every word and return the final output!

```python
def transform(self, text: str):
    self._ensure_fitted()

    output_ids = []
    for word in self.normalize_text(text).split():
        word_tokens = self._byte_tokenize(word)

        # Apply learned merges in the same order they were created
        for pair in self.merges:
            word_tokens = self._merge_pair(word_tokens, pair, self.merge_to_id[pair])

        output_ids.extend(word_tokens)

    return output_ids
```

Note that since we split on whitespace before tokenizing, the spaces themselves are dropped: decoding the ids reconstructs the normalized text without its spaces. Production tokenizers keep spaces around by attaching them to the start of words (that's what the funny `Ġ` symbol means if you've ever peeked inside GPT-2's vocabulary!), but we keep things simple here.

## `WordPiece`

BPE isn't the only subword tokenizer out there. **WordPiece** is the algorithm behind BERT and its whole family, and at first glance it looks *very* similar to BPE: start from small pieces, iteratively merge pairs into bigger vocabulary entries. The interesting part is in the differences!

<div align="center">
  <img src="../../../images/tokenizers/WordPiece.gif" alt="WordPiece learning character pieces with continuation prefixes and tokenizing with greedy longest-match" width="850">
</div>

**Difference #1: characters and the `##` prefix.** WordPiece starts from characters instead of bytes, and it marks every piece that *continues* a word with a special prefix, `##` by default. The word "playing" starts its life as `["p", "##l", "##a", "##y", "##i", "##n", "##g"]`:

```python
def create_splits_and_word_counts(self, words_corpus):
    word_counts = {}
    splits = {}

    for word in words_corpus:
        word_counts[word] = word_counts.get(word, 0) + 1
        splits[word] = [word[0]]

        for character in word[1:]:
            splits[word].append(f"{self.continuation_prefix}{character}")

    return word_counts, splits
```

Why bother with the prefix? Because "ing" at the start of a word (like in "ingest") and "ing" as a suffix (like in "playing") are very different things! With the prefix, they become two distinct tokens ("ing" vs "##ing"), and the model gets to learn different meanings for each.

**Difference #2: the merge criterion.** BPE merges the most *frequent* pair. WordPiece instead merges the pair with the highest **score**:

$$score = \frac{freq(pair)}{freq(first) \times freq(second)}$$

```python
def compute_scores(self, word_counts, splits):
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
```

The intuition: dividing by the frequencies of the individual pieces *penalizes* pairs whose parts are already everywhere on their own. A pair like ("t", "##h") might be the most frequent one in an English corpus, but "t" and "##h" also show up in a million other places, so its score stays low. Meanwhile, a pair whose two pieces almost only ever appear *together* gets a high score, even if it's rarer. In other words: BPE merges what is **common**, WordPiece merges what is **strongly associated**.

### Training WordPiece

With those two ingredients, training looks a lot like BPE's loop, just with scores instead of raw counts:

```python
def fit(self, corpus: str):
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
```

When two pieces merge, we glue their strings together and drop the continuation prefix of the second one, so ("play", "##ing") becomes "playing":

```python
def pair_to_token(self, pair):
    return pair[0] + pair[1].removeprefix(self.continuation_prefix)
```

### Transforming with WordPiece

**Difference #3: how we tokenize new text.** Remember how BPE replays every learnt merge, in order? WordPiece throws the merge history away entirely! It only keeps the final vocabulary, and tokenizes each word by **greedy longest-match**: find the longest piece in the vocabulary that matches the start of the word, emit it, and continue from where it ended.

```python
def _tokenize_word(self, word: str):
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
```

Notice how harsh that middle part is: if *any* part of a word cannot be matched, the **whole word** becomes `<unk>`. That's not a bug, real BERT does exactly the same! And since WordPiece starts from characters instead of bytes, a single unseen character is all it takes. Score another point for byte-level BPE.

The `transform()` method then just runs this greedy matching on every word and looks up the ids:

```python
def transform(self, text: str):
    self._ensure_fitted()

    output_ids = []
    unknown_token_id = self.token_to_id[self.unk_token]

    for word in self.normalize_text(text).split():
        word_tokens = self._tokenize_word(word)
        for token in word_tokens:
            output_ids.append(self.token_to_id.get(token, unknown_token_id))

    return output_ids
```

## Example Usage

Let's see our tokenizers in action:

```python
from smolml.preprocessing.tokenizers import CharacterTokenizer, WordTokenizer, BPETokenizer, WordPiece

corpus = "the cat sat on the mat while the other cats watched the birds"

# Character-level: tiny vocab, long sequences
char_tok = CharacterTokenizer()
char_ids = char_tok.fit_transform(corpus)
print(f"Chars -> vocab size: {len(char_tok.token_to_id)}, sequence length: {len(char_ids)}")

# Word-level: short sequences, big vocab (and <unk> for new words!)
word_tok = WordTokenizer()
word_ids = word_tok.fit_transform(corpus)
print(f"Words -> vocab size: {len(word_tok.token_to_id)}, sequence length: {len(word_ids)}")
print("Unseen word:", word_tok.decode(word_tok.transform("the dog sat")))

# BPE: the middle ground
bpe_tok = BPETokenizer(num_merges=20)
bpe_ids = bpe_tok.fit_transform(corpus)
print(f"BPE -> sequence length: {len(bpe_ids)}")
print("Learned pieces:", bpe_tok.decode(bpe_ids))

# WordPiece: greedy longest-match with ## continuations
wp_tok = WordPiece(iterations=30)
wp_ids = wp_tok.fit_transform(corpus)
print(f"WordPiece -> vocab size: {len(wp_tok.token_to_id)}, sequence length: {len(wp_ids)}")
print("Learned pieces:", wp_tok.decode(wp_ids))
```

Try playing with `num_merges`: with 0 merges you get plain bytes (long sequences), and the more merges you allow, the shorter the sequences get, at the price of a bigger vocabulary. That's the whole tradeoff of tokenization in a single parameter!

## Run the tests!

You can check out how these tokenizers behave by running `tokenizers.py` in the `tests/` folder! It compares our implementations against standard frameworks and prints a report of token pieces, ids, and vocabulary sizes.

[Next Section - Activation Functions](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils/activation)

## Resources & Readings

- [Hugging Face LLM Course - Tokenizers](https://huggingface.co/learn/llm-course/chapter6/1)
- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Sennrich et al. - Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (the paper that brought BPE to NLP)
- [Schuster & Nakajima - Japanese and Korean Voice Search](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) (where WordPiece was born)
- [Tiktokenizer](https://tiktokenizer.vercel.app/) (play with real LLM tokenizers in your browser!)
