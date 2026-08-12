"""
train.py — BLaIR Dual Encoder Training
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data

sys.path.insert(0, ".")
from src.dataset import RetrievalDataset, collate_fn, build_hard_negatives_bm25
from src.encoder import build_encoder
from src.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",     type=str,   default="data/")
    p.add_argument("--output-dir",   type=str,   default="artifacts/models/")
    p.add_argument("--model-type",   type=str,   default="dual",
                   choices=["biencoder","dual"])
    p.add_argument("--bert-model",   type=str,   default="bert-base-uncased")
    p.add_argument("--pooling",      type=str,   default="mean",
                   choices=["mean","cls"])
    p.add_argument("--max-len",      type=int,   default=128)
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--temperature",  type=float, default=0.05)
    p.add_argument("--neg-mode",     type=str,   default="random",
                   choices=["random","bm25"])
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("\n" + "="*60)
    print("BLaIR Dense Retrieval Training")
    print("="*60)
    for k, v in vars(args).items():
        print(f"  {k:<15}: {v}")
    print(f"  device         : {device}")

    print("\nLoading data...")
    train_df  = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    corpus_df = pd.read_parquet(os.path.join(args.data_dir, "corpus.parquet"))
    print(f"  Train pairs : {len(train_df):,}")
    print(f"  Corpus size : {len(corpus_df):,}")

    corpus_ids       = corpus_df["product_id"].tolist()
    corpus_docs      = corpus_df["product_doc"].tolist()
    corpus_id_to_doc = dict(zip(corpus_ids, corpus_docs))

    print(f"\nBuilding {args.model_type} encoder ({args.bert_model})...")
    model = build_encoder(
        model_type = args.model_type,
        model_name = args.bert_model,
        pooling    = args.pooling,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    hard_negatives = None
    if args.neg_mode == "bm25":
        print("\nMining BM25 hard negatives (k=3)...")
        hard_negatives = build_hard_negatives_bm25(
            train_df=train_df, corpus_df=corpus_df, k=3)

    print("\nBuilding dataset and dataloader...")
    dataset = RetrievalDataset(
        df               = train_df,
        corpus_docs      = corpus_docs,
        corpus_id_to_doc = corpus_id_to_doc,
        tokenizer        = model.tokenizer,
        max_len          = args.max_len,
        mode             = args.neg_mode,
        hard_negatives   = hard_negatives,
        seed             = args.seed,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = min(args.batch_size, 16),
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        collate_fn  = collate_fn(model.tokenizer, max_len=args.max_len),
        drop_last   = True,
    )
    print(f"  Dataset: {len(dataset):,} | Batches/epoch: {len(dataloader):,}")

    trainer = Trainer(
        model        = model,
        train_loader = dataloader,
        args         = args,
        output_dir   = args.output_dir,
    )

    print("\nStarting training...")
    start   = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved -> {hist_path}")

    print(f"\nTraining complete in {elapsed/60:.1f} min")
    for ep, (loss, t) in enumerate(
            zip(history["train_loss"], history["epoch_times"]), 1):
        print(f"  Epoch {ep}: loss={loss:.4f}  time={t:.1f}s")


if __name__ == "__main__":
    main()
