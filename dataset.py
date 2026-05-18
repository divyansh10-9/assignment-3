"""
dataset.py — Multi30k Dataset Loader and Vocabulary Builder
DA6401 Assignment 3: Transformer for Machine Translation
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import Counter
import spacy


# ════════════════════════════════════════════════════════════════
# Vocabulary Class
# ════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple vocabulary wrapper.

    Special Tokens:
        <unk> : Unknown token
        <pad> : Padding token
        <sos> : Start-of-sentence token
        <eos> : End-of-sentence token
    """

    def __init__(self, freq_threshold: int = 2):

        self.freq_threshold = freq_threshold

        # index → token
        self.itos = {
            0: "<unk>",
            1: "<pad>",
            2: "<sos>",
            3: "<eos>",
        }

        # token → index
        self.stoi = {
            "<unk>": 0,
            "<pad>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }

    def __len__(self):
        return len(self.itos)

    # ------------------------------------------------------------

    @staticmethod
    def tokenizer_de(text, spacy_de):
        """
        German tokenizer using spaCy.
        """
        return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

    @staticmethod
    def tokenizer_en(text, spacy_en):
        """
        English tokenizer using spaCy.
        """
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    # ------------------------------------------------------------

    def build_vocabulary(self, sentence_list, tokenizer):
        """
        Build vocabulary from list of sentences.
        """

        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:

            for word in tokenizer(sentence):

                frequencies[word] += 1

                # add word when threshold reached
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    # ------------------------------------------------------------

    def numericalize(self, text, tokenizer):
        """
        Convert sentence to list of token indices.
        """

        tokenized_text = tokenizer(text)

        return [
            self.stoi[token]
            if token in self.stoi
            else self.stoi["<unk>"]
            for token in tokenized_text
        ]


# ════════════════════════════════════════════════════════════════
# Multi30k Dataset
# ════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):

    def __init__(
        self,
        split='train',
        src_vocab=None,
        tgt_vocab=None,
    ):
        """
        Loads the Multi30k dataset and prepares tokenizers.

        Args:
            split     : 'train', 'validation', or 'test'
            src_vocab : Optional pre-built source (German) Vocabulary.
                        If provided, this vocabulary is reused instead
                        of building a new one from the current split.
                        REQUIRED for val/test splits so that token
                        indices are consistent with the training vocab.
            tgt_vocab : Optional pre-built target (English) Vocabulary.
                        Same semantics as src_vocab.

        BUG FIX: accept pre-built vocabularies for val/test splits.
        Previously every split built its own vocabulary, giving val
        and test much smaller vocabs with *different* token→index
        mappings than the training vocab.  The model was built with
        train-vocab dimensions, so val/test sequences indexed the
        wrong embeddings — validation loss and BLEU were meaningless.
        Now train passes its built vocabs into the val/test datasets.
        """

        super().__init__()

        self.split = split

        print(f"\nLoading Multi30k {split} split...")

        # --------------------------------------------------------
        # Load dataset from HuggingFace
        # https://huggingface.co/datasets/bentrevett/multi30k
        # --------------------------------------------------------

        self.dataset = load_dataset(
            "bentrevett/multi30k",
            split=split
        )

        # --------------------------------------------------------
        # Load spaCy tokenizers
        # --------------------------------------------------------

        # try:
        #     self.spacy_de = spacy.load("de_core_news_sm")
        # except:
        #     raise RuntimeError(
        #         "German spaCy model not found.\n"
        #         "Run:\n"
        #         "python -m spacy download de_core_news_sm"
        #     )

        # try:
        #     self.spacy_en = spacy.load("en_core_web_sm")
        # except:
        #     raise RuntimeError(
        #         "English spaCy model not found.\n"
        #         "Run:\n"
        #         "python -m spacy download en_core_web_sm"
        #     )
        self.spacy_de = spacy.blank("de")
        self.spacy_en = spacy.blank("en")

        # --------------------------------------------------------
        # Build or reuse vocabularies
        # --------------------------------------------------------

        if src_vocab is not None and tgt_vocab is not None:
            # Reuse caller-supplied vocabularies (val / test splits)
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            print(
                f"Reusing supplied vocabularies "
                f"(German: {len(self.src_vocab)}, "
                f"English: {len(self.tgt_vocab)})"
            )
        else:
            # Build fresh vocabularies from this split (train only)
            self.src_vocab = Vocabulary(freq_threshold=2)
            self.tgt_vocab = Vocabulary(freq_threshold=2)
            self.build_vocab()

        # --------------------------------------------------------
        # Process tokenized numericalized data
        # --------------------------------------------------------

        self.data = self.process_data()

        print(f"Loaded {len(self.data)} samples.\n")

    # ════════════════════════════════════════════════════════════

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (German)
        and tgt (English), including:
            <unk>, <pad>, <sos>, <eos>
        """

        print("Building vocabularies...")

        german_sentences = [item["de"] for item in self.dataset]
        english_sentences = [item["en"] for item in self.dataset]

        self.src_vocab.build_vocabulary(
            german_sentences,
            lambda text: Vocabulary.tokenizer_de(
                text,
                self.spacy_de
            )
        )

        self.tgt_vocab.build_vocabulary(
            english_sentences,
            lambda text: Vocabulary.tokenizer_en(
                text,
                self.spacy_en
            )
        )

        print(f"German Vocabulary Size : {len(self.src_vocab)}")
        print(f"English Vocabulary Size: {len(self.tgt_vocab)}")

    # ════════════════════════════════════════════════════════════

    def process_data(self):
        """
        Convert English and German sentences into integer
        token lists using spaCy and vocabularies.
        """

        print("Processing dataset...")

        processed_data = []

        for item in self.dataset:

            src_sentence = item["de"]
            tgt_sentence = item["en"]

            # ----------------------------------------------------
            # Numericalize German sentence
            # ----------------------------------------------------

            src_indices = [self.src_vocab.stoi["<sos>"]]

            src_indices += self.src_vocab.numericalize(
                src_sentence,
                lambda text: Vocabulary.tokenizer_de(
                    text,
                    self.spacy_de
                )
            )

            src_indices.append(
                self.src_vocab.stoi["<eos>"]
            )

            # ----------------------------------------------------
            # Numericalize English sentence
            # ----------------------------------------------------

            tgt_indices = [self.tgt_vocab.stoi["<sos>"]]

            tgt_indices += self.tgt_vocab.numericalize(
                tgt_sentence,
                lambda text: Vocabulary.tokenizer_en(
                    text,
                    self.spacy_en
                )
            )

            tgt_indices.append(
                self.tgt_vocab.stoi["<eos>"]
            )

            # ----------------------------------------------------
            # Convert to tensors
            # ----------------------------------------------------

            src_tensor = torch.tensor(
                src_indices,
                dtype=torch.long
            )

            tgt_tensor = torch.tensor(
                tgt_indices,
                dtype=torch.long
            )

            processed_data.append(
                (src_tensor, tgt_tensor)
            )

        return processed_data

    # ════════════════════════════════════════════════════════════

    def __len__(self):
        return len(self.data)

    # ════════════════════════════════════════════════════════════

    def __getitem__(self, idx):
        return self.data[idx]


# ════════════════════════════════════════════════════════════════
# Collate Function
# ════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_idx=1):
    """
    Pads variable-length sequences in a batch.

    Args:
        batch : List[(src_tensor, tgt_tensor)]

    Returns:
        src_batch : [batch_size, src_len]
        tgt_batch : [batch_size, tgt_len]
    """

    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_idx
    )

    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_idx
    )

    return src_batch, tgt_batch


# ════════════════════════════════════════════════════════════════
# Quick Test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    dataset = Multi30kDataset(split='train')

    print("Dataset Size:", len(dataset))

    src, tgt = dataset[0]

    print("\nGerman Tensor:")
    print(src)

    print("\nEnglish Tensor:")
    print(tgt)

    print("\nGerman Vocabulary Size:")
    print(len(dataset.src_vocab))

    print("\nEnglish Vocabulary Size:")
    print(len(dataset.tgt_vocab))
