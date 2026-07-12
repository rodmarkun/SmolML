import hashlib
import os
import sys
import textwrap
import unittest
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import CountVectorizer

from smolml.preprocessing.tokenizers import (
    BPETokenizer,
    CharacterTokenizer,
    NotFittedError,
    WordPiece,
    WordTokenizer,
)
from tests.img_utils import image_path

"""
/////////////////////
/// TOKENIZER TESTS ///
/////////////////////

Comparison tests for the SmolML tokenizer implementations.
Each test builds a SmolML tokenizer, compares it against a standard framework
where a direct comparison is useful, and prints a console report for review.
"""


@dataclass
class TokenizerComparison:
    """
    Stores one tokenizer comparison row for the console report.
    Keeps tokens, ids, vocabulary sizes, and notes together for printing.
    """
    name: str
    framework: str
    input_text: str
    smol_tokens: list
    framework_tokens: list
    smol_ids: list
    framework_ids: list
    smol_vocab_size: int | None
    framework_vocab_size: int | None
    note: str = ""


class TestTokenizerComparison(unittest.TestCase):
    """
    Compare SmolML tokenizers against standard tokenization frameworks and
    print a console report of token pieces, ids, and vocabulary sizes.
    """
    def setUp(self):
        """
        Sets up a small repeated corpus so subword tokenizers have enough
        examples to learn useful pieces.
        """
        self.corpus = (
            "Small models learn from small data. "
            "Small tokenizers split words into tokens. "
            "Words and characters become ids for models. "
            "Small tokenizers split words into tokens again."
        )
        self.input_text = "Small tokenizers split words into tokens."

    def _format_token(self, token):
        """
        Converts special visual cases into readable labels for console output.
        """
        if token == " ":
            return "<space>"
        if token == "":
            return "<empty>"
        return str(token)

    def _format_sequence(self, values, max_items=18):
        """
        Formats long token/id sequences so they fit inside the console report.
        """
        formatted = [self._format_token(value) for value in values]
        if len(formatted) > max_items:
            formatted = (
                formatted[:max_items]
                + [f"... +{len(values) - max_items} more"]
            )
        return " | ".join(formatted)

    def _sklearn_comparison(self, tokenizer, vectorizer, name, framework):
        """
        Runs a SmolML-vs-scikit-learn comparison: both should split the
        normalized input identically, and SmolML ids should match sklearn ids
        offset by 2 (SmolML reserves <pad> and <unk>). Returns the report row.
        """
        tokenizer.fit(self.corpus)

        normalized_corpus = tokenizer.normalize_text(self.corpus)
        normalized_input = tokenizer.normalize_text(self.input_text)
        vectorizer.fit([normalized_corpus])

        smol_ids = tokenizer.transform(self.input_text)
        smol_tokens = tokenizer.decode(smol_ids)
        sklearn_tokens = vectorizer.build_analyzer()(normalized_input)
        sklearn_ids = [vectorizer.vocabulary_[token] for token in sklearn_tokens]

        self.assertEqual(smol_tokens, sklearn_tokens)
        self.assertEqual(smol_ids, [token_id + 2 for token_id in sklearn_ids])

        return TokenizerComparison(
            name=name,
            framework=framework,
            input_text=normalized_input,
            smol_tokens=smol_tokens,
            framework_tokens=sklearn_tokens,
            smol_ids=smol_ids,
            framework_ids=sklearn_ids,
            smol_vocab_size=len(tokenizer.token_to_id),
            framework_vocab_size=len(vectorizer.vocabulary_),
            note="SmolML ids are offset by 2 because <pad> and <unk> are reserved.",
        )

    def _sklearn_character_comparison(self):
        """
        Compares CharacterTokenizer against scikit-learn character analysis.
        """
        return self._sklearn_comparison(
            CharacterTokenizer(),
            CountVectorizer(analyzer='char', lowercase=False),
            "CharacterTokenizer",
            "scikit-learn CountVectorizer(analyzer='char')",
        )

    def _sklearn_word_comparison(self):
        """
        Compares WordTokenizer against scikit-learn word analysis.
        """
        return self._sklearn_comparison(
            WordTokenizer(),
            CountVectorizer(
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b',
                lowercase=False,
            ),
            "WordTokenizer",
            "scikit-learn CountVectorizer(analyzer='word')",
        )

    def _import_huggingface_tokenizers(self):
        """
        This test file is named tokenizers.py, which would otherwise shadow the
        Hugging Face tokenizers package when the test is run directly.
        """
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        original_path = list(sys.path)
        shadowed_module = sys.modules.get("tokenizers")
        shadowed_module_file = getattr(shadowed_module, "__file__", "") or ""

        try:
            if (
                shadowed_module is not None
                and shadowed_module_file.startswith(tests_dir)
            ):
                del sys.modules["tokenizers"]

            sys.path = [
                path for path in sys.path
                if os.path.abspath(path or os.getcwd()) != tests_dir
            ]

            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers import models, pre_tokenizers, trainers
        finally:
            sys.path = original_path
            if shadowed_module is not None:
                sys.modules["tokenizers"] = shadowed_module

        return HFTokenizer, models, pre_tokenizers, trainers

    def _huggingface_bpe_comparison(self):
        """
        Compares BPETokenizer against Hugging Face ByteLevel BPE.
        Both pre-tokenize by word so merges never cross word boundaries, but
        exact ids, byte display, and tie-breaking still differ, so the test
        checks structural properties and prints both learned segmentations.
        """
        try:
            (
                HFTokenizer,
                models,
                pre_tokenizers,
                trainers,
            ) = self._import_huggingface_tokenizers()
        except ImportError:
            self.skipTest(
                "Hugging Face tokenizers is required for the BPE comparison. "
                "Install tests/requirements.txt before running this test."
            )

        tokenizer = BPETokenizer(num_merges=15, min_frequency=2)
        tokenizer.fit(self.corpus)

        normalized_corpus = tokenizer.normalize_text(self.corpus)
        normalized_input = tokenizer.normalize_text(self.input_text)

        hf_tokenizer = HFTokenizer(models.BPE(unk_token="<unk>"))
        hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=tokenizer.byte_tokenization_size + tokenizer.num_merges + 1,
            min_frequency=tokenizer.min_frequency,
            special_tokens=["<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        hf_tokenizer.train_from_iterator([normalized_corpus], trainer=trainer)

        smol_ids = tokenizer.transform(self.input_text)
        smol_tokens = tokenizer.decode(smol_ids)
        hf_encoding = hf_tokenizer.encode(normalized_input)

        # Word pre-tokenization drops spaces, so the pieces reconstruct the
        # normalized input with its spaces removed.
        expected_text = normalized_input.replace(" ", "")

        self.assertGreater(len(tokenizer.merges), 0)
        self.assertGreater(len(smol_ids), 0)
        self.assertGreater(len(hf_encoding.ids), 0)
        self.assertEqual("".join(smol_tokens), expected_text)

        return TokenizerComparison(
            name="BPETokenizer",
            framework="Hugging Face tokenizers ByteLevel BPE",
            input_text=normalized_input,
            smol_tokens=smol_tokens,
            framework_tokens=hf_encoding.tokens,
            smol_ids=smol_ids,
            framework_ids=hf_encoding.ids,
            smol_vocab_size=len(tokenizer.id_to_token),
            framework_vocab_size=hf_tokenizer.get_vocab_size(),
            note=(
                "Both pre-tokenize by word, but BPE implementations do not "
                "share ids, byte display, or merge tie-breaking; the report "
                "compares the learned segmentation shape."
            ),
        )

    def _strip_wordpiece_prefixes(self, tokens, continuation_prefix="##"):
        """
        Removes WordPiece continuation prefixes so token pieces can be joined
        back into the original word characters.
        """
        return "".join(
            token.removeprefix(continuation_prefix)
            for token in tokens
            if token != "<unk>"
        )

    def _huggingface_wordpiece_comparison(self):
        """
        Compares WordPiece against a Hugging Face WordPiece trained with its own
        WordPieceTrainer. The HF trainer learns its vocabulary with a
        frequency-based BPE rule while SmolML follows the published
        likelihood-score rule, so the learned vocabularies, tokens, and ids
        differ. Both use greedy longest-match segmentation and reconstruct the
        input, so the test checks those structural properties and prints both
        segmentations rather than asserting equality.
        """
        try:
            (
                HFTokenizer,
                models,
                pre_tokenizers,
                trainers,
            ) = self._import_huggingface_tokenizers()
        except ImportError:
            self.skipTest(
                "Hugging Face tokenizers is required for the WordPiece comparison. "
                "Install tests/requirements.txt before running this test."
            )

        tokenizer = WordPiece(iterations=30)
        tokenizer.fit(self.corpus)

        normalized_corpus = tokenizer.normalize_text(self.corpus)
        normalized_input = tokenizer.normalize_text(self.input_text)

        hf_tokenizer = HFTokenizer(
            models.WordPiece(
                unk_token=tokenizer.unk_token,
                continuing_subword_prefix=tokenizer.continuation_prefix,
            )
        )
        hf_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordPieceTrainer(
            vocab_size=len(tokenizer.token_to_id),
            min_frequency=1,
            special_tokens=[tokenizer.pad_token, tokenizer.unk_token],
            continuing_subword_prefix=tokenizer.continuation_prefix,
        )
        hf_tokenizer.train_from_iterator([normalized_corpus], trainer=trainer)

        smol_ids = tokenizer.transform(self.input_text)
        smol_tokens = tokenizer.decode(smol_ids)
        hf_encoding = hf_tokenizer.encode(normalized_input)

        expected_text = normalized_input.replace(" ", "")
        smol_text = self._strip_wordpiece_prefixes(
            smol_tokens,
            tokenizer.continuation_prefix,
        )
        hf_text = self._strip_wordpiece_prefixes(
            hf_encoding.tokens,
            tokenizer.continuation_prefix,
        )

        self.assertGreater(len(smol_ids), 0)
        self.assertGreater(len(hf_encoding.ids), 0)
        self.assertNotIn(tokenizer.unk_token, smol_tokens)
        self.assertEqual(smol_text, expected_text)
        self.assertEqual(hf_text, expected_text)

        return TokenizerComparison(
            name="WordPiece",
            framework="Hugging Face tokenizers WordPiece (WordPieceTrainer)",
            input_text=normalized_input,
            smol_tokens=smol_tokens,
            framework_tokens=hf_encoding.tokens,
            smol_ids=smol_ids,
            framework_ids=hf_encoding.ids,
            smol_vocab_size=len(tokenizer.token_to_id),
            framework_vocab_size=hf_tokenizer.get_vocab_size(),
            note=(
                "Both segment greedily and reconstruct the input, but the HF "
                "WordPieceTrainer learns its vocabulary with a frequency-based "
                "BPE rule while SmolML uses the likelihood-score rule, so "
                "vocabularies, tokens, and ids differ."
            ),
        )

    def _print_wrapped(self, label, value, width=120):
        """
        Prints a labeled line and wraps long values with aligned indentation.
        """
        line = f"{label}: {value}"
        print(
            textwrap.fill(
                line,
                width=width,
                subsequent_indent=" " * (len(label) + 2),
            )
        )

    def _print_comparisons(self, comparisons):
        """
        Prints a compact report showing tokens, ids, and vocabulary sizes.
        """
        print("\nTokenizer Comparison Report")
        print("=" * 80)

        for comparison in comparisons:
            title = f"{comparison.name} vs {comparison.framework}"
            print(f"\n{title}")
            print("-" * len(title))

            # Normalized input text
            self._print_wrapped("Input", comparison.input_text)

            # Token sequences from SmolML and the comparison framework
            self._print_wrapped(
                f"SmolML tokens ({len(comparison.smol_tokens)})",
                self._format_sequence(comparison.smol_tokens),
            )
            self._print_wrapped(
                f"Framework tokens ({len(comparison.framework_tokens)})",
                self._format_sequence(comparison.framework_tokens),
            )

            # Token id sequences from SmolML and the comparison framework
            self._print_wrapped(
                "SmolML ids",
                self._format_sequence(comparison.smol_ids),
            )
            self._print_wrapped(
                "Framework ids",
                self._format_sequence(comparison.framework_ids),
            )

            self._print_wrapped(
                "Vocab sizes",
                (
                    f"SmolML={comparison.smol_vocab_size}, "
                    f"framework={comparison.framework_vocab_size}"
                ),
            )
            self._print_wrapped("Note", comparison.note)

    def _display_token(self, token):
        """
        Converts a token into its plot label, making spaces visible.
        """
        display = str(token).replace(" ", "␣")
        return display if display else "<empty>"

    def _token_color(self, display_token):
        """
        Picks a deterministic palette color per unique token text so the same
        piece gets the same color in every strip of the plot.
        """
        palette = [
            "#f94144", "#f3722c", "#f8961e", "#f9c74f",
            "#90be6d", "#43aa8b", "#4d908e", "#577590",
            "#277da1", "#9b5de5", "#f15bb5", "#00bbf9",
            "#00f5d4", "#fee440", "#8ac926", "#ff99c8",
        ]
        digest = hashlib.md5(display_token.encode("utf-8")).hexdigest()
        return palette[int(digest, 16) % len(palette)]

    def _plot_token_strip(self, ax, tokens, y, strip_height):
        """
        Draws one tokenization as a strip of rounded blocks sized by token
        length and returns the total strip width in data units.
        """
        block_gap = 0.25
        block_pad = 0.8
        x = 0.0

        for token in tokens:
            display = self._display_token(token)
            width = len(display) + block_pad

            ax.add_patch(
                FancyBboxPatch(
                    (x, y - strip_height / 2),
                    width - block_gap,
                    strip_height,
                    boxstyle="round,pad=0,rounding_size=0.14",
                    facecolor=self._token_color(display),
                    edgecolor="#0e1117",
                    linewidth=0.6,
                )
            )
            ax.text(
                x + (width - block_gap) / 2,
                y,
                display,
                ha="center",
                va="center",
                fontsize=7.5,
                family="monospace",
                color="#101319",
            )
            x += width

        return x

    def _plot_comparisons(self, comparisons):
        """
        Renders every comparison as paired segmentation strips on a dark
        dashboard. Block width is proportional to token length and identical
        pieces share a color, so granularity differences and SmolML/reference
        agreement are visible at a glance. Returns the saved image path.
        """
        output_path = image_path("tokenizer_comparison.png")
        background = "#0e1117"
        strip_height = 0.72

        fig, ax = plt.subplots(figsize=(16, 9.5))
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)

        y = 0.0
        max_x = 0.0

        for comparison in comparisons:
            ax.text(
                0,
                y,
                f"{comparison.name}  vs  {comparison.framework}",
                color="#e8eaf0",
                fontsize=11,
                fontweight="bold",
                va="center",
            )
            y -= 1.15

            strips = (
                ("SmolML", comparison.smol_tokens, comparison.smol_vocab_size),
                (
                    "reference",
                    comparison.framework_tokens,
                    comparison.framework_vocab_size,
                ),
            )
            for label, tokens, vocab_size in strips:
                strip_width = self._plot_token_strip(ax, tokens, y, strip_height)

                ax.text(
                    -1.0,
                    y,
                    label,
                    ha="right",
                    va="center",
                    fontsize=9.5,
                    color="#8b93a7",
                )
                ax.text(
                    strip_width + 0.8,
                    y,
                    f"{len(tokens)} tokens · vocab {vocab_size}",
                    va="center",
                    fontsize=8.5,
                    color="#8b93a7",
                )

                max_x = max(max_x, strip_width)
                y -= 1.05

            y -= 0.85

        ax.set_xlim(-9, max_x + 18)
        ax.set_ylim(y, 1.6)
        ax.axis("off")

        fig.text(
            0.5,
            0.975,
            "SmolML Tokenizer Gallery — one sentence, four ways to slice it",
            ha="center",
            color="#e8eaf0",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.945,
            f'input: "{comparisons[0].input_text}"   ·   '
            "block width ∝ token length   ·   same piece → same color",
            ha="center",
            color="#8b93a7",
            fontsize=10,
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=background,
        )
        plt.close(fig)

        return output_path

    def test_tokenizers_against_standard_frameworks(self):
        """
        Run every currently available tokenizer comparison and print a single
        console report for visual inspection.
        """
        comparisons = [
            self._sklearn_character_comparison(),
            self._sklearn_word_comparison(),
            self._huggingface_bpe_comparison(),
            self._huggingface_wordpiece_comparison(),
        ]

        self._print_comparisons(comparisons)
        output_path = self._plot_comparisons(comparisons)
        print(f"\nSaved tokenizer comparison image to {output_path}")

        self.assertEqual(len(comparisons), 4)


class TestTokenizerApi(unittest.TestCase):
    """
    Framework-free checks for the shared tokenizer API: Unicode-safe byte
    tokenization, decode round-trips, the fit_transform/tokenize helpers, and
    the not-fitted guard.
    """
    def setUp(self):
        """
        Sets up a small corpus reused across the API checks.
        """
        self.corpus = (
            "Small models learn from small data. "
            "Small tokenizers split words into tokens."
        )
        self.input_text = "Small tokenizers split words into tokens."
        self.tokenizer_types = (
            CharacterTokenizer,
            WordTokenizer,
            BPETokenizer,
            WordPiece,
        )

    def test_bpe_handles_non_latin1_unicode(self):
        """
        BPE must tokenize and round-trip characters outside latin-1. The old
        latin-1 byte encoding raised UnicodeEncodeError on these, so this
        guards against that regression.
        """
        text = "coffee ☕ café smörgåsbord"
        tokenizer = BPETokenizer(num_merges=10)
        tokenizer.fit(text)

        token_ids = tokenizer.transform(text)

        # id_to_token stores each UTF-8 byte as a latin-1 display character, so
        # joining the decoded tokens and re-decoding recovers the input bytes.
        # Word pre-tokenization drops spaces, so compare against spaceless text.
        recovered = (
            "".join(tokenizer.decode(token_ids)).encode("latin-1").decode("utf-8")
        )
        expected = tokenizer.normalize_text(text).replace(" ", "")

        self.assertGreater(len(token_ids), 0)
        self.assertEqual(recovered, expected)

    def test_character_decode_round_trips(self):
        """
        Decoding CharacterTokenizer ids returns the normalized characters.
        """
        tokenizer = CharacterTokenizer()
        tokenizer.fit(self.corpus)

        normalized = tokenizer.normalize_text(self.corpus)
        recovered = "".join(tokenizer.decode(tokenizer.transform(self.corpus)))

        self.assertEqual(recovered, normalized)

    def test_word_decode_round_trips(self):
        """
        Decoding WordTokenizer ids returns the normalized whitespace words.
        """
        tokenizer = WordTokenizer()
        tokenizer.fit(self.corpus)

        normalized = tokenizer.normalize_text(self.corpus)
        recovered = " ".join(tokenizer.decode(tokenizer.transform(self.corpus)))

        self.assertEqual(recovered, normalized)

    def test_fit_transform_matches_fit_then_transform(self):
        """
        fit_transform must equal calling fit then transform on the corpus for
        every tokenizer, since all now share the fit(corpus) signature.
        """
        for tokenizer_type in self.tokenizer_types:
            with self.subTest(tokenizer=tokenizer_type.__name__):
                combined = tokenizer_type().fit_transform(self.corpus)

                separate = tokenizer_type()
                separate.fit(self.corpus)
                expected = separate.transform(self.corpus)

                self.assertEqual(combined, expected)

    def test_tokenize_is_transform_alias(self):
        """
        tokenize must return the same ids as transform for every tokenizer.
        """
        for tokenizer_type in self.tokenizer_types:
            with self.subTest(tokenizer=tokenizer_type.__name__):
                tokenizer = tokenizer_type()
                tokenizer.fit(self.corpus)

                self.assertEqual(
                    tokenizer.tokenize(self.input_text),
                    tokenizer.transform(self.input_text),
                )

    def test_transform_before_fit_raises(self):
        """
        Transforming before fit must raise NotFittedError for every tokenizer.
        """
        for tokenizer_type in self.tokenizer_types:
            with self.subTest(tokenizer=tokenizer_type.__name__):
                with self.assertRaises(NotFittedError):
                    tokenizer_type().transform("small tokens")


if __name__ == '__main__':
    unittest.main()
