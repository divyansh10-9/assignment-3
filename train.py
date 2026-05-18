"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from typing import Optional
from functools import partial

import wandb

from dataset import Multi30kDataset, collate_fn
from model import (
    Transformer,
    make_src_mask,
    make_tgt_mask
)
from lr_scheduler import NoamScheduler


# ════════════════════════════════════════════════════════════════
# Label Smoothing Loss
# ════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        smoothing: float = 0.1
    ) -> None:

        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    # ------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:

        logits = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():

            true_dist = torch.zeros_like(logits)

            true_dist.fill_(
                self.smoothing / (self.vocab_size - 2)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                self.confidence
            )

            true_dist[:, self.pad_idx] = 0

            mask = (target == self.pad_idx)

            true_dist[mask] = 0

        loss = torch.mean(
            torch.sum(-true_dist * logits, dim=-1)
        )

        return loss


# ════════════════════════════════════════════════════════════════
# Run Epoch
# ════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    global_step: int = 0,
) -> tuple:
    """
    Run one epoch of training or validation.

    Returns:
        avg_loss   : Average loss over the epoch
        global_step: Updated global step counter (only incremented during training)
    """

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss  = 0
    total_tokens = 0   # non-pad tokens seen (for perplexity)

    progress_bar = tqdm(
        data_iter,
        desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}"
    )

    for batch_idx, batch in enumerate(progress_bar):

        src = batch[0].to(device)
        tgt = batch[1].to(device)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        # count non-pad tokens for perplexity
        non_pad = (tgt_output != 1).sum().item()
        total_tokens += non_pad

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):

            logits = model(src, tgt_input, src_mask, tgt_mask)

            logits     = logits.reshape(-1, logits.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = loss_fn(logits, tgt_output)

            if is_train:

                loss.backward()

                # track gradient norm before clipping
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                global_step += 1

                # ── per-step wandb logging ────────────────────────
                wandb.log({
                    "train/step_loss"  : loss.item(),
                    "train/learning_rate": current_lr,
                    "train/grad_norm"  : grad_norm,
                    "global_step"      : global_step,
                })

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}" if is_train else ""
        )

    avg_loss = total_loss / len(data_iter)

    # perplexity = exp(avg token-level NLL)
    # avg_loss here is already mean over *positions* inside the loss fn,
    # so we recompute a token-level estimate from total_loss * batch_count
    perplexity = math.exp(min(avg_loss, 20))   # cap to avoid overflow

    return avg_loss, perplexity, global_step


# ════════════════════════════════════════════════════════════════
# Greedy Decoding
# ════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:

    model.eval()

    src      = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).to(device)

    for _ in range(max_len - 1):

        tgt_mask = make_tgt_mask(ys)

        out = model.decode(memory, src_mask, ys, tgt_mask)

        prob = out[:, -1]

        _, next_word = torch.max(prob, dim=1)

        next_word = next_word.item()

        ys = torch.cat(
            [
                ys,
                torch.ones(1, 1, dtype=torch.long).fill_(next_word).to(device)
            ],
            dim=1
        )

        if next_word == end_symbol:
            break

    return ys


# ════════════════════════════════════════════════════════════════
# BLEU Evaluation
# ════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
    log_n_examples: int = 5,
) -> float:
    """
    Compute corpus BLEU on test_dataloader.

    Args:
        log_n_examples : Number of translation examples to log to wandb.

    Returns:
        BLEU score (0–100)
    """

    model.eval()

    references  = []
    hypotheses  = []
    examples    = []   # (src_str, ref_str, hyp_str) for wandb table

    sos_idx = tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]

    with torch.no_grad():

        for batch in tqdm(test_dataloader, desc="BLEU eval"):

            src = batch[0].to(device)
            tgt = batch[1].to(device)

            for i in range(src.shape[0]):

                src_sample = src[i].unsqueeze(0)
                src_mask   = make_src_mask(src_sample)

                prediction = greedy_decode(
                    model,
                    src_sample,
                    src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device
                )

                pred_tokens = prediction.squeeze(0).tolist()

                pred_sentence = []
                for idx in pred_tokens:
                    token = tgt_vocab.itos[idx]
                    if token == "<eos>":
                        break
                    if token not in ["<sos>", "<pad>"]:
                        pred_sentence.append(token)

                tgt_tokens = tgt[i].tolist()
                tgt_sentence = []
                for idx in tgt_tokens:
                    token = tgt_vocab.itos[idx]
                    if token == "<eos>":
                        break
                    if token not in ["<sos>", "<pad>"]:
                        tgt_sentence.append(token)

                hypotheses.append(pred_sentence)
                references.append([tgt_sentence])

                # collect a few examples for the wandb table
                if len(examples) < log_n_examples:
                    examples.append((
                        " ".join(tgt_sentence),          # reference (EN)
                        " ".join(pred_sentence),         # hypothesis (EN)
                    ))

    bleu = corpus_bleu(
        references,
        hypotheses,
        smoothing_function=SmoothingFunction().method1,
    ) * 100

    # log translation examples table to wandb
    table = wandb.Table(
        columns=["reference", "hypothesis"],
        data=examples,
    )
    wandb.log({"translations/examples": table})

    return bleu


# ════════════════════════════════════════════════════════════════
# Save Checkpoint
# ════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "model_config": {

            "src_vocab_size":
                model.src_embedding.num_embeddings,

            "tgt_vocab_size":
                model.tgt_embedding.num_embeddings,

            "d_model":
                model.d_model,
        }
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


# ════════════════════════════════════════════════════════════════
# Load Checkpoint
# ════════════════════════════════════════════════════════════════

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:

    checkpoint = torch.load(path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from {path}")

    return checkpoint["epoch"]


# ════════════════════════════════════════════════════════════════
# Training Experiment
# ════════════════════════════════════════════════════════════════

def run_training_experiment():

    config = {
        "batch_size"   : 32,
        "d_model"      : 512,
        "num_heads"    : 8,
        "num_layers"   : 6,
        "d_ff"         : 2048,
        "dropout"      : 0.1,
        "epochs"       : 10,
        "warmup_steps" : 4000,
        "lr"           : 1.0,
    }

    wandb.init(
        project="da6401-a3",
        config=config,
        tags=["transformer", "de-en", "multi30k"],
    )

    # log model architecture summary as text
    wandb.run.notes = (
        f"Transformer {config['num_layers']}L × {config['num_heads']}H, "
        f"d_model={config['d_model']}, d_ff={config['d_ff']}, "
        f"warmup={config['warmup_steps']}"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    wandb.config.update({"device": str(device)})

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------

    train_dataset = Multi30kDataset("train")

    val_dataset = Multi30kDataset(
        "validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    test_dataset = Multi30kDataset(
        "test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    # log vocab sizes
    wandb.config.update({
        "src_vocab_size": len(train_dataset.src_vocab),
        "tgt_vocab_size": len(train_dataset.tgt_vocab),
        "train_samples" : len(train_dataset),
        "val_samples"   : len(val_dataset),
        "test_samples"  : len(test_dataset),
    })

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    # log parameter count
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({
        "total_params"    : total_params,
        "trainable_params": trainable_params,
    })
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")

    # watch gradients and weights (logs histograms every 100 steps)
    wandb.watch(model, log="all", log_freq=100)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )

    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=1,
        smoothing=0.1,
    )

    # ------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------

    best_val_loss  = float("inf")
    best_val_epoch = 0
    global_step    = 0

    for epoch in range(config["epochs"]):

        epoch_start = time.time()

        train_loss, train_ppl, global_step = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            global_step=global_step,
        )

        val_loss, val_ppl, _ = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            global_step=global_step,
        )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch}"
            f"\n  Train Loss: {train_loss:.4f}  PPL: {train_ppl:.2f}"
            f"\n  Val   Loss: {val_loss:.4f}  PPL: {val_ppl:.2f}"
            f"\n  Time: {epoch_time:.1f}s\n"
        )

        # ── per-epoch wandb logging ───────────────────────────────
        wandb.log({
            "epoch"            : epoch,
            "train/epoch_loss" : train_loss,
            "train/perplexity" : train_ppl,
            "val/epoch_loss"   : val_loss,
            "val/perplexity"   : val_ppl,
            "epoch_time_sec"   : epoch_time,
            "global_step"      : global_step,
        })

        if val_loss < best_val_loss:

            best_val_loss  = val_loss
            best_val_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path="best_model.pt",
            )

            # save best model as a wandb artifact
            artifact = wandb.Artifact(
                name="best_model",
                type="model",
                description=f"Best checkpoint at epoch {epoch}, val_loss={val_loss:.4f}",
            )
            artifact.add_file("best_model.pt")
            wandb.log_artifact(artifact)

            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

    wandb.run.summary["best_val_loss"]  = best_val_loss
    wandb.run.summary["best_val_epoch"] = best_val_epoch

    # ------------------------------------------------------------
    # BLEU Evaluation on Test Set
    # ------------------------------------------------------------

    # load best weights before evaluating
    load_checkpoint("best_model.pt", model)

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_vocab,
        device=device,
        log_n_examples=10,
    )

    print(f"\nFinal BLEU Score: {bleu:.2f}")

    wandb.log({"test/bleu": bleu})
    wandb.run.summary["test_bleu"] = bleu

    wandb.finish()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    run_training_experiment()
