"""
src/encoder.py
==============
BiEncoder (shared weights) and DualEncoder (separate weights, BLaIR-style)
built on bert-base-uncased with mean-pooling and L2 normalization.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer, AutoTokenizer


class MeanPooling(nn.Module):
    """
    Mean pool BERT last hidden states with attention mask weighting.

    Computes the weighted mean of token embeddings, ignoring padding tokens.
    This is more stable than CLS pooling for asymmetric retrieval tasks.
    """

    def forward(
        self,
        last_hidden_state: torch.Tensor,   # (B, T, H)
        attention_mask: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:                      # (B, H)
        # Expand mask to (B, T, H) for broadcasting
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Sum masked embeddings
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        # Sum of actual (non-padding) tokens per example
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class CLSPooling(nn.Module):
    """CLS token pooling — used in ablation B."""

    def forward(
        self,
        last_hidden_state: torch.Tensor,   # (B, T, H)
        attention_mask: torch.Tensor,       # (B, T)  [unused but kept for interface]
    ) -> torch.Tensor:                      # (B, H)
        return last_hidden_state[:, 0, :]   # CLS is always position 0


def _make_pooling(pooling: str) -> nn.Module:
    if pooling == "cls":
        return CLSPooling()
    return MeanPooling()


class BiEncoder(nn.Module):
    """
    Shared-weight bi-encoder (Sentence-BERT style).

    A single BERT instance encodes both review queries and product documents.
    This is the simplest baseline — both modalities share the same 110M
    parameter space.

    Architecture:
        text_encoder = BertModel("bert-base-uncased")
        item_encoder = text_encoder  # SAME INSTANCE
        Both encode with mean pooling → 768-dim L2-normalized embedding.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_len: int = 128,
        normalize: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.normalize = normalize
        self.pooling_type = pooling

        self.bert = BertModel.from_pretrained(model_name or "bert-base-uncased")
        self.pooler = _make_pooling(pooling)
        # Shared: both query and doc go through self.bert
        _tok_name = model_name or "bert-base-uncased"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(_tok_name, use_fast=False)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

    def _encode_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch with shared BERT and pool → (B, 768)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.pooler(out.last_hidden_state, attention_mask)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(
        self,
        query_input_ids: torch.Tensor,   # (B, T)
        query_attn_mask: torch.Tensor,   # (B, T)
        doc_input_ids: torch.Tensor,     # (B, T)
        doc_attn_mask: torch.Tensor,     # (B, T)
    ):
        """
        Returns:
            query_emb : (B, 768) L2-normalized
            doc_emb   : (B, 768) L2-normalized
        """
        query_emb = self._encode_single(query_input_ids, query_attn_mask)
        doc_emb   = self._encode_single(doc_input_ids,   doc_attn_mask)
        return query_emb, doc_emb

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """General encode method for any list of texts."""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eval()
        self.to(device)
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = self._encode_single(enc["input_ids"], enc["attention_mask"])
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    @torch.no_grad()
    def encode_queries(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Encode review query texts. Larger batch ok (eval mode)."""
        return self.encode(texts, batch_size=batch_size, device=device)

    @torch.no_grad()
    def encode_docs(
        self,
        texts: List[str],
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode product document texts.
        batch_size=8 ALWAYS for corpus encoding (BERT is large).
        """
        return self.encode(texts, batch_size=batch_size, device=device)

    def save(self, path: str):
        """Save BERT weights, tokenizer, and config."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.bert.save_pretrained(os.path.join(path, "bert"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        config = {
            "model_type": "biencoder",
            "model_name": self.model_name,
            "max_len": self.max_len,
            "normalize": self.normalize,
            "pooling": self.pooling_type,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"[BiEncoder] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "BiEncoder":
        """Load from saved checkpoint."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(
            model_name=os.path.join(path, "bert"),
            max_len=config["max_len"],
            normalize=config["normalize"],
            pooling=config.get("pooling", "mean"),
        )
        # Overwrite tokenizer with saved one
        model.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "tokenizer")
        )
        if device is not None:
            model = model.to(device)
        print(f"[BiEncoder] Loaded from {path}")
        return model

    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable()


class DualEncoder(nn.Module):
    """
    Separate-weight dual encoder (BLaIR-style).

    Two independent BERT instances — one for review queries (text_encoder)
    and one for product documents (item_encoder). Both are initialized from
    bert-base-uncased but have SEPARATE parameters that are trained jointly.

    Key scientific insight:
        Review language is colloquial, experiential, subjective.
        Product language is technical, feature-based, structured.
        Separate encoders can learn specialized representations for each
        modality without forcing them through a shared representational
        bottleneck. This mirrors what Amazon BLaIR does at scale.

    Architecture:
        text_encoder = BertModel("bert-base-uncased")   ← query tower
        item_encoder = BertModel("bert-base-uncased")   ← product tower (SEPARATE)
        Both encode with mean pooling → 768-dim L2-normalized embedding.
        Total parameters: 2x bert-base-uncased ≈ 220M
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_len: int = 128,
        normalize: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_len = max_len
        self.normalize = normalize
        self.pooling_type = pooling

        # Two independent BERT instances
        self.text_encoder = BertModel.from_pretrained(model_name or "bert-base-uncased")  # query tower
        self.item_encoder = BertModel.from_pretrained(model_name or "bert-base-uncased")  # product tower

        self.query_pooler = _make_pooling(pooling)
        self.doc_pooler   = _make_pooling(pooling)

        _tok_name = model_name or "bert-base-uncased"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(_tok_name, use_fast=False)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

    def _encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.query_pooler(out.last_hidden_state, attention_mask)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def _encode_doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.item_encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.doc_pooler(out.last_hidden_state, attention_mask)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def forward(
        self,
        query_input_ids: torch.Tensor,   # (B, T)
        query_attn_mask: torch.Tensor,   # (B, T)
        doc_input_ids: torch.Tensor,     # (B, T)
        doc_attn_mask: torch.Tensor,     # (B, T)
    ):
        """
        Returns:
            query_emb : (B, 768) L2-normalized, from text_encoder
            doc_emb   : (B, 768) L2-normalized, from item_encoder
        """
        query_emb = self._encode_query(query_input_ids, query_attn_mask)
        doc_emb   = self._encode_doc(doc_input_ids, doc_attn_mask)
        return query_emb, doc_emb

    @torch.no_grad()
    def encode_queries(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Encode review texts with text_encoder (query tower)."""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eval()
        self.to(device)
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = self._encode_query(enc["input_ids"], enc["attention_mask"])
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    @torch.no_grad()
    def encode_docs(
        self,
        texts: List[str],
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode product documents with item_encoder (product tower).
        batch_size=8 ALWAYS for corpus encoding on T4 GPU.
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eval()
        self.to(device)
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Encoding corpus", unit="batch"):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            emb = self._encode_doc(enc["input_ids"], enc["attention_mask"])
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    def save(self, path: str):
        """Save both encoder weights, tokenizer, and config."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.text_encoder.save_pretrained(os.path.join(path, "text_encoder"))
        self.item_encoder.save_pretrained(os.path.join(path, "item_encoder"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        config = {
            "model_type": "dual",
            "model_name": self.model_name,
            "max_len": self.max_len,
            "normalize": self.normalize,
            "pooling": self.pooling_type,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"[DualEncoder] Saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DualEncoder":
        """Load from saved checkpoint."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.model_name = config["model_name"]
        model.max_len    = config["max_len"]
        model.normalize  = config["normalize"]
        model.pooling_type = config.get("pooling", "mean")

        model.text_encoder = BertModel.from_pretrained(
            os.path.join(path, "text_encoder")
        )
        model.item_encoder = BertModel.from_pretrained(
            os.path.join(path, "item_encoder")
        )
        model.query_pooler = _make_pooling(model.pooling_type)
        model.doc_pooler   = _make_pooling(model.pooling_type)
        model.tokenizer    = AutoTokenizer.from_pretrained(
            os.path.join(path, "tokenizer")
        )
        if device is not None:
            model = model.to(device)
        print(f"[DualEncoder] Loaded from {path}")
        return model

    def gradient_checkpointing_enable(self):
        self.text_encoder.gradient_checkpointing_enable()
        self.item_encoder.gradient_checkpointing_enable()


def build_encoder(
    model_type: str,
    model_name: str = "bert-base-uncased",
    max_len: int = 128,
    pooling: str = "mean",
) -> nn.Module:
    """Factory function. model_type: 'biencoder' or 'dual'."""
    if model_type == "biencoder":
        return BiEncoder(model_name=model_name, max_len=max_len, pooling=pooling)
    elif model_type == "dual":
        return DualEncoder(model_name=model_name, max_len=max_len, pooling=pooling)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'biencoder' or 'dual'.")


def load_encoder(path: str) -> nn.Module:
    """Load either BiEncoder or DualEncoder from checkpoint."""
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    if config["model_type"] == "biencoder":
        return BiEncoder.load(path)
    elif config["model_type"] == "dual":
        return DualEncoder.load(path)
    else:
        raise ValueError(f"Unknown model_type in config: {config['model_type']}")
