"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import copy
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
# Scaled Dot Product Attention
# ════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q,
    K,
    V,
    mask=None,
    dropout: Optional[nn.Dropout] = None,
):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    BUG FIX: added `dropout` parameter.
    The original paper (Section 3.2.2) and the original code in
    `MultiHeadAttention` both defined a Dropout layer but the
    dropout was never actually applied to the attention weights.
    Dropout is now applied after the softmax, before the weighted
    sum with V, exactly as described in the paper.
    """

    d_k = Q.size(-1)

    scores = torch.matmul(
        Q,
        K.transpose(-2, -1)
    ) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

    attention_weights = torch.softmax(
        scores,
        dim=-1
    )

    # BUG FIX: apply dropout to attention weights
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(
        attention_weights,
        V
    )

    return output, attention_weights


# ════════════════════════════════════════════════════════════════
# Source Mask
# ════════════════════════════════════════════════════════════════

def make_src_mask(src, pad_idx=1):
    """
    Returns True at padding positions (these will be masked out
    by masked_fill → -inf → 0 weight after softmax).

    Shape: [batch, 1, 1, src_len]  — broadcasts over heads and
    query positions in the encoder self-attention and the
    decoder cross-attention.
    """

    return (
        src == pad_idx
    ).unsqueeze(1).unsqueeze(2)


# ════════════════════════════════════════════════════════════════
# Target Mask
# ════════════════════════════════════════════════════════════════

def make_tgt_mask(tgt, pad_idx=1):
    """
    Combines padding mask and causal (look-ahead) mask.

    Shape: [batch, 1, tgt_len, tgt_len]
    """

    batch_size, tgt_len = tgt.shape

    pad_mask = (
        tgt == pad_idx
    ).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones(
            tgt_len,
            tgt_len,
            device=tgt.device
        ),
        diagonal=1
    ).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    return pad_mask | causal_mask


# ════════════════════════════════════════════════════════════════
# Multi Head Attention
# ════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.1
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        K = K.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        V = V.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        # BUG FIX: pass self.dropout so attention weights are
        # regularised during training (was defined but never used)
        attention_output, _ = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask,
            dropout=self.dropout,
        )

        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(attention_output)

        return output


# ════════════════════════════════════════════════════════════════
# Positional Encoding
# ════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    # ------------------------------------------------------------

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


# ════════════════════════════════════════════════════════════════
# Feed Forward Network
# ════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------

    def forward(self, x):

        return self.linear2(
            self.dropout(
                F.relu(self.linear1(x))
            )
        )


# ════════════════════════════════════════════════════════════════
# Encoder Layer
# ════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------

    def forward(self, x, src_mask):

        attn_output = self.self_attn(x, x, x, src_mask)

        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)

        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ════════════════════════════════════════════════════════════════
# Decoder Layer
# ════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------

    def forward(self, x, memory, src_mask, tgt_mask):

        self_attn_output = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(
            x, memory, memory, src_mask
        )

        x = self.norm2(x + self.dropout(cross_attn_output))

        ffn_output = self.ffn(x)

        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ════════════════════════════════════════════════════════════════
# Encoder
# ════════════════════════════════════════════════════════════════

class Encoder(nn.Module):

    def __init__(self, layer, N):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ════════════════════════════════════════════════════════════════
# Decoder
# ════════════════════════════════════════════════════════════════

class Decoder(nn.Module):

    def __init__(self, layer, N):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


# ════════════════════════════════════════════════════════════════
# Full Transformer
# ════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Transformer model for German → English translation.

    On instantiation (i.e., inside __init__), the model:
        1. Loads spaCy tokenizers (de_core_news_sm, en_core_web_sm)
        2. Builds / loads source and target vocabularies from the
           Multi30k training split
        3. Downloads pretrained weights from Google Drive via gdown
           (if not already cached locally) and loads them

    This means Transformer() with all defaults is a fully
    ready-to-use translation model — no external setup required.

    Args:
        src_vocab_size  : German vocabulary size  (set from loaded vocab)
        tgt_vocab_size  : English vocabulary size (set from loaded vocab)
        d_model         : Embedding / hidden dimension
        N               : Number of encoder/decoder layers
        num_heads       : Number of attention heads
        d_ff            : Feed-forward inner dimension
        dropout         : Dropout probability
        weight_path     : Local path where weights are cached
        gdrive_file_id  : Google Drive file-id for the weights
                          (set this to your own Drive file id before
                          submission; leave empty to skip download)
        device          : 'cuda', 'cpu', or None (auto-detect)
        max_infer_len   : Maximum tokens to generate during inference
    """

    # ── ❶ replace this with your actual Google Drive file ID ──
    _DEFAULT_GDRIVE_FILE_ID = "1lGcq7RlC9rE618LGza8G4r8yzENhyQmD"
    _DEFAULT_WEIGHT_PATH    = "best_model.pt"

    def __init__(
        self,
        src_vocab_size: int = 0,          # filled from loaded vocab
        tgt_vocab_size: int = 0,          # filled from loaded vocab
        d_model:   int = 512,
        N:         int = 6,
        num_heads: int = 8,
        d_ff:      int = 2048,
        dropout:   float = 0.1,
        weight_path:    str = _DEFAULT_WEIGHT_PATH,
        gdrive_file_id: str = _DEFAULT_GDRIVE_FILE_ID,
        device: Optional[str] = None,
        max_infer_len: int = 100,
    ):
        # ── device ───────────────────────────────────────────────
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        self._max_infer_len = max_infer_len

        # ── ❷ load spaCy tokenizers ───────────────────────────────
        import spacy

        try:
            self._spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            raise RuntimeError(
                "German spaCy model not found.\n"
                "Run:  python -m spacy download de_core_news_sm"
            )

        try:
            self._spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "English spaCy model not found.\n"
                "Run:  python -m spacy download en_core_web_sm"
            )

        # ── ❸ build vocabularies from Multi30k train split ────────
        from dataset import Multi30kDataset

        print("Loading vocabularies from Multi30k training data...")
        _train_data = Multi30kDataset(split="train")

        self.src_vocab = _train_data.src_vocab
        self.tgt_vocab = _train_data.tgt_vocab

        # Override vocab sizes from the actual loaded vocabularies
        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)

        # ── ❹ build model architecture ────────────────────────────
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # ── ❺ download weights (if needed) and load them ─────────
        self._load_weights(
            weight_path=weight_path,
            gdrive_file_id=gdrive_file_id,
        )

        self.to(self._device)

    # ════════════════════════════════════════════════════════════
    # Weight loading helper
    # ════════════════════════════════════════════════════════════

    def _load_weights(
        self,
        weight_path: str,
        gdrive_file_id: str,
    ) -> None:
        """
        Download weights from Google Drive (if not cached locally)
        and load them into this model.
        """

        # ── download if the file doesn't exist yet ────────────────
        if not os.path.exists(weight_path):

            if gdrive_file_id and gdrive_file_id != "YOUR_GDRIVE_FILE_ID_HERE":
                import gdown
                print(
                    f"Downloading weights from Google Drive "
                    f"(file id: {gdrive_file_id}) → {weight_path} ..."
                )
                url = f"https://drive.google.com/uc?id={gdrive_file_id}"
                gdown.download(url, weight_path, quiet=False)
            else:
                print(
                    f"[WARNING] Weight file '{weight_path}' not found "
                    f"and no valid gdrive_file_id provided. "
                    f"Model will use random weights."
                )
                return

        # ── load checkpoint ───────────────────────────────────────
        print(f"Loading weights from '{weight_path}' ...")

        checkpoint = torch.load(
            weight_path,
            map_location=self._device,
            weights_only=False,
        )

        # Checkpoints saved by train.py wrap state_dict under the
        # 'model_state_dict' key; handle both formats gracefully.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict)

        print("Weights loaded successfully.")

    # ════════════════════════════════════════════════════════════
    # Tokenizer helpers (mirrors dataset.py static methods)
    # ════════════════════════════════════════════════════════════

    def _tokenize_de(self, text: str):
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def _tokenize_en_ids(self, token_ids) -> str:
        """Convert a list of tgt vocab indices back to an English string."""
        tokens = []
        for idx in token_ids:
            token = self.tgt_vocab.itos.get(idx, "<unk>")
            if token == "<eos>":
                break
            if token not in ("<sos>", "<pad>", "<unk>"):
                tokens.append(token)
        return " ".join(tokens)

    # ════════════════════════════════════════════════════════════
    # encode / decode / forward  (unchanged from original)
    # ════════════════════════════════════════════════════════════

    def encode(self, src, src_mask):

        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)

        return self.encoder(src, src_mask)

    # ------------------------------------------------------------

    def decode(self, memory, src_mask, tgt, tgt_mask):

        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)

        output = self.decoder(tgt, memory, src_mask, tgt_mask)

        return self.fc_out(output)

    # ------------------------------------------------------------

    def forward(self, src, tgt, src_mask, tgt_mask):

        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)

        return output

    # ════════════════════════════════════════════════════════════
    # infer  — end-to-end German → English
    # ════════════════════════════════════════════════════════════

    def infer(self, src_sentence: str) -> str:
        """
        Translate a single German sentence to English.

        Pipeline:
            1. Tokenize the German input with spaCy
            2. Numericalize using the source vocabulary
               (wrap with <sos> / <eos>)
            3. Build source mask and run the encoder
            4. Autoregressively decode (greedy) until <eos>
               or max_infer_len tokens are produced
            5. Convert predicted token indices back to an
               English string and return it

        Args:
            src_sentence : Raw German string, e.g.
                           "Ein Mann sitzt auf einer Bank."

        Returns:
            Predicted English translation as a plain string.
        """

        self.eval()

        with torch.no_grad():

            # ── Step 1 & 2 : tokenize + numericalize ─────────────

            tokens = self._tokenize_de(src_sentence)

            src_indices = (
                [self.src_vocab.stoi["<sos>"]]
                + [
                    self.src_vocab.stoi.get(tok, self.src_vocab.stoi["<unk>"])
                    for tok in tokens
                ]
                + [self.src_vocab.stoi["<eos>"]]
            )

            src_tensor = torch.tensor(
                src_indices,
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(0)   # [1, src_len]

            # ── Step 3 : encode ───────────────────────────────────

            src_mask = make_src_mask(src_tensor)           # [1,1,1,src_len]
            memory   = self.encode(src_tensor, src_mask)   # [1, src_len, d_model]

            # ── Step 4 : autoregressive greedy decoding ───────────

            sos_idx = self.tgt_vocab.stoi["<sos>"]
            eos_idx = self.tgt_vocab.stoi["<eos>"]

            # Start with just the <sos> token
            ys = torch.tensor(
                [[sos_idx]],
                dtype=torch.long,
                device=self._device,
            )

            for _ in range(self._max_infer_len - 1):

                tgt_mask = make_tgt_mask(ys)

                out  = self.decode(memory, src_mask, ys, tgt_mask)
                prob = out[:, -1]                         # [1, tgt_vocab_size]

                _, next_word = torch.max(prob, dim=1)
                next_word    = next_word.item()

                ys = torch.cat(
                    [
                        ys,
                        torch.tensor(
                            [[next_word]],
                            dtype=torch.long,
                            device=self._device,
                        ),
                    ],
                    dim=1,
                )

                if next_word == eos_idx:
                    break

            # ── Step 5 : detokenize ───────────────────────────────

            predicted_ids = ys.squeeze(0).tolist()
            english_sentence = self._tokenize_en_ids(predicted_ids)

        return english_sentence
