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
        
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:

            for word in tokenizer(sentence):

                frequencies[word] += 1

                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1


    def numericalize(self, text, tokenizer):
        
        tokenized_text = tokenizer(text)

        return [
            self.stoi[token]
            if token in self.stoi
            else self.stoi["<unk>"]
            for token in tokenized_text
        ]



class Multi30kDataset(Dataset):

    def __init__(
        self,
        split='train',
        src_vocab=None,
        tgt_vocab=None,
    ):
       

        super().__init__()

        self.split = split

        print(f"\nLoading Multi30k {split} split...")

       
        self.dataset = load_dataset(
            "bentrevett/multi30k",
            split=split
        )

        self.spacy_de = spacy.blank("de")
        self.spacy_en = spacy.blank("en")

       
        if src_vocab is not None and tgt_vocab is not None:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            print(
                f"Reusing supplied vocabularies "
                f"(German: {len(self.src_vocab)}, "
                f"English: {len(self.tgt_vocab)})"
            )
        else:
            self.src_vocab = Vocabulary(freq_threshold=2)
            self.tgt_vocab = Vocabulary(freq_threshold=2)
            self.build_vocab()

        self.data = self.process_data()

        print(f"Loaded {len(self.data)} samples.\n")


    def build_vocab(self):
        

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

   
    def process_data(self):
       
        print("Processing dataset...")

        processed_data = []

        for item in self.dataset:

            src_sentence = item["de"]
            tgt_sentence = item["en"]

            
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

    
    def __len__(self):
        return len(self.data)

    

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, pad_idx=1):
    
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
