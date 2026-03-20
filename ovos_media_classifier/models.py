"""StaticModelForHierarchicalClassification — inference-ready model class.

This module contains the two-headed Model2Vec classifier used for OCP intent
classification.  It is the production home of the class that was previously
defined in the training script (ovos-m2v-pipeline/train/train_hierarchical.py).

Architecture::

    sentence → StaticModel encoder → text_emb (D)
                                            │
                            ┌───────────────┴───────────────────────┐
                            │                                       │
                      domain_head(emb)     intent_head(cat(emb, softmax(domain_logits)))
                            │                                       │
                      domain_logits (n_domains)             intent_logits (n_intents)
                                                    (masked to domain intents at inference)

Training loss:  L = L_domain + lambda_intent * L_intent
Inference:
  1. domain_pred  = argmax(domain_logits)
  2. Mask intent_logits: zero out intents NOT in the predicted domain
  3. intent_pred  = argmax(masked_intent_logits)

Dependencies:
  Inference only:  torch, model2vec
  Training:        + lightning, scikit-learn, tqdm

Install the full training stack with::

    pip install ovos-media-classifier[train]
"""
from __future__ import annotations

import logging
import os
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from model2vec import StaticModel
from model2vec.train.base import FinetunableStaticModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (can be overridden via fit() kwargs or from_pretrained() kwargs)
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_LAYERS: int = 1
HIDDEN_DIM: int = 256
FREEZE_EMBEDDINGS: bool = False


# ---------------------------------------------------------------------------
# Dataset helper (needed for both training and loading)
# ---------------------------------------------------------------------------

class HierarchicalTextDataset(Dataset):
    """Tokenised dataset for the two-headed hierarchical model."""

    def __init__(
        self,
        tokenized_texts: list[list[int]],
        domain_labels: torch.Tensor,
        intent_labels: torch.Tensor,
    ) -> None:
        assert len(tokenized_texts) == len(domain_labels) == len(intent_labels)
        self.tokenized_texts = tokenized_texts
        self.domain_labels = domain_labels
        self.intent_labels = intent_labels

    def __len__(self) -> int:
        return len(self.tokenized_texts)

    def __getitem__(self, index: int):
        return (
            self.tokenized_texts[index],
            self.domain_labels[index],
            self.intent_labels[index],
        )

    @staticmethod
    def collate_fn(batch):
        texts, domain_labels, intent_labels = zip(*batch)
        tensors = [torch.LongTensor(t) for t in texts]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)
        return padded, torch.stack(domain_labels), torch.stack(intent_labels)

    def to_dataloader(self, shuffle: bool, batch_size: int = 32) -> DataLoader:
        return DataLoader(
            self,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StaticModelForHierarchicalClassification(FinetunableStaticModel):
    """Two-headed hierarchical classifier on a StaticModel encoder.

    Head 1 (domain_head): text_emb → domain_logits
    Head 2 (intent_head): cat(text_emb, softmax(domain_logits)) → intent_logits

    At inference intent logits are masked to the intents that belong to the
    predicted domain (domain_intent_mask built from training data).

    This class is the production version of the identical class that was
    previously defined in the training repository.  The ``fit()`` method
    requires ``lightning``; the ``predict*`` and ``save_pretrained`` /
    ``from_pretrained`` methods work with ``torch`` only.
    """

    def __init__(
        self,
        *,
        vectors: torch.Tensor,
        tokenizer,
        n_domains: int,
        n_intents: int,
        n_layers: int = N_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        pad_id: int = 0,
        token_mapping: list[int] | None = None,
        weights: torch.Tensor | None = None,
        freeze: bool = FREEZE_EMBEDDINGS,
    ) -> None:
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_domains = n_domains
        self.n_intents = n_intents
        self.domain_classes_: list[str] = [str(i) for i in range(n_domains)]
        self.intent_classes_: list[str] = [str(i) for i in range(n_intents)]

        super().__init__(
            vectors=vectors,
            out_dim=n_domains,
            pad_id=pad_id,
            tokenizer=tokenizer,
            token_mapping=token_mapping,
            weights=weights,
            freeze=freeze,
        )

        self.intent_head: nn.Sequential = self._build_head(
            in_dim=self.embed_dim + n_domains,
            out_dim=n_intents,
        )
        self.register_buffer(
            "domain_intent_mask",
            torch.zeros(n_domains, n_intents, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    # Head construction
    # ------------------------------------------------------------------

    def construct_head(self) -> nn.Sequential:
        return self._build_head(in_dim=self.embed_dim, out_dim=self.n_domains)

    def _build_head(self, in_dim: int, out_dim: int) -> nn.Sequential:
        if self.n_layers == 0:
            modules: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        else:
            modules = [nn.Linear(in_dim, self.hidden_dim), nn.ReLU()]
            for _ in range(self.n_layers - 1):
                modules += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
            modules.append(nn.Linear(self.hidden_dim, out_dim))
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        *initial, last = linear_layers
        for m in initial:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(last.weight)
        nn.init.zeros_(last.bias)
        return nn.Sequential(*modules)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor):
        emb = self._encode(input_ids)
        domain_logits = self.head(emb)
        domain_soft = torch.softmax(domain_logits, dim=-1)
        intent_logits = self.intent_head(torch.cat([emb, domain_soft], dim=-1))
        return (domain_logits, intent_logits), emb

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_batch(self, texts: list[str]):
        ids = self.tokenize(texts).to(self.device)
        (d_logits, i_logits), _ = self.forward(ids)
        return d_logits, i_logits

    def predict(
        self,
        X: list[str],
        batch_size: int = 1024,
        show_progress_bar: bool = False,
        mask_intents: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict domain and intent for each text. Returns (domains, intents)."""
        try:
            from tqdm import trange
        except ImportError:
            def trange(n, **_kw):  # minimal fallback
                return range(n)

        self.eval()
        d_preds, i_preds = [], []
        for start in trange(
            0, len(X), batch_size, disable=not show_progress_bar
        ):
            d_logits, i_logits = self._forward_batch(X[start: start + batch_size])
            d_pred = d_logits.argmax(dim=1)
            if mask_intents:
                i_logits = i_logits.masked_fill(
                    ~self.domain_intent_mask[d_pred], float("-inf")
                )
            d_preds.extend(d_pred.tolist())
            i_preds.extend(i_logits.argmax(dim=1).tolist())
        return (
            np.array([self.domain_classes_[i] for i in d_preds]),
            np.array([self.intent_classes_[i] for i in i_preds]),
        )

    def predict_proba(
        self, X: list[str], batch_size: int = 1024, mask_intents: bool = True
    ):
        """Return (domain_probs, intent_probs) as numpy arrays."""
        self.eval()
        d_list, i_list = [], []
        for start in range(0, len(X), batch_size):
            d_logits, i_logits = self._forward_batch(X[start: start + batch_size])
            d_pred = d_logits.argmax(dim=1)
            if mask_intents:
                i_logits = i_logits.masked_fill(
                    ~self.domain_intent_mask[d_pred], float("-inf")
                )
            d_list.append(torch.softmax(d_logits, dim=1).cpu().numpy())
            i_list.append(torch.softmax(i_logits, dim=1).cpu().numpy())
        return np.concatenate(d_list), np.concatenate(i_list)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: list[str],
        y_domain: list[str],
        y_intent: list[str],
        learning_rate: float = 1e-3,
        batch_size: int | None = None,
        max_epochs: int = -1,
        min_epochs: int | None = None,
        early_stopping_patience: int | None = 5,
        test_size: float = 0.1,
        device: str = "auto",
        X_val: list[str] | None = None,
        y_domain_val: list[str] | None = None,
        y_intent_val: list[str] | None = None,
        lambda_intent: float = 1.0,
    ) -> "StaticModelForHierarchicalClassification":
        """Train both heads jointly on (utterance, domain, intent) triples.

        Requires ``lightning`` and ``scikit-learn``::

            pip install ovos-media-classifier[train]
        """
        try:
            import lightning as pl
            from lightning.pytorch.callbacks import EarlyStopping
        except ImportError:
            raise ImportError(
                "Training requires 'lightning'. "
                "Install it with: pip install ovos-media-classifier[train]"
            )
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "Training requires 'scikit-learn'. "
                "Install it with: pip install ovos-media-classifier[train]"
            )

        pl.seed_everything(RANDOM_STATE)
        self._initialize_classes(y_domain, y_intent)

        if X_val is not None:
            if y_domain_val is None or y_intent_val is None:
                raise ValueError("Provide y_domain_val and y_intent_val with X_val.")
            train_X, train_yd, train_yi = X, y_domain, y_intent
            val_X, val_yd, val_yi = X_val, y_domain_val, y_intent_val
        else:
            train_X, val_X, train_yd, val_yd, train_yi, val_yi = train_test_split(
                X, y_domain, y_intent,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=y_domain,
            )

        if batch_size is None:
            base = int(min(max(1, len(train_X) / 30 // 32), 16))
            batch_size = int(base * 32)
            logger.info("Auto batch size: %d", batch_size)

        train_ds = self._prepare_dataset(train_X, train_yd, train_yi)
        val_ds = self._prepare_dataset(val_X, val_yd, val_yi)

        module = _HierarchicalLightningModule(
            self, learning_rate=learning_rate, lambda_intent=lambda_intent
        )

        n_train_batches = len(train_ds) // batch_size
        callbacks = []
        if early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_intent_acc",
                    mode="max",
                    patience=early_stopping_patience,
                )
            )

        if n_train_batches < 250:
            val_check_interval = None
            check_val_every_epoch = 1
        else:
            val_check_interval = max(250, 2 * len(val_ds) // batch_size)
            check_val_every_epoch = None

        with TemporaryDirectory() as tmpdir:
            trainer = pl.Trainer(
                min_epochs=min_epochs,
                max_epochs=max_epochs,
                callbacks=callbacks,
                val_check_interval=val_check_interval,
                check_val_every_n_epoch=check_val_every_epoch,
                accelerator=device,
                default_root_dir=tmpdir,
            )
            trainer.fit(
                module,
                train_dataloaders=train_ds.to_dataloader(
                    shuffle=True, batch_size=batch_size
                ),
                val_dataloaders=val_ds.to_dataloader(
                    shuffle=False, batch_size=batch_size
                ),
            )
            best_path = trainer.checkpoint_callback.best_model_path
            best_weights = torch.load(best_path, weights_only=True)

        state_dict = {
            k.removeprefix("model."): v
            for k, v in best_weights["state_dict"].items()
            if "loss_function" not in k
        }
        self.load_state_dict(state_dict)
        self.eval()
        return self

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _initialize_classes(
        self, y_domain: list[str], y_intent: list[str]
    ) -> None:
        self.domain_classes_ = sorted(set(y_domain))
        self.intent_classes_ = sorted(set(y_intent))
        self.n_domains = len(self.domain_classes_)
        self.n_intents = len(self.intent_classes_)
        self.out_dim = self.n_domains
        self.head = self.construct_head()
        self.intent_head = self._build_head(
            in_dim=self.embed_dim + self.n_domains,
            out_dim=self.n_intents,
        )
        self.embeddings = nn.Embedding.from_pretrained(
            self.vectors.clone(), freeze=self.freeze, padding_idx=self.pad_id
        )
        self.w = self.construct_weights()

        d_idx = {d: i for i, d in enumerate(self.domain_classes_)}
        i_idx = {i: j for j, i in enumerate(self.intent_classes_)}
        mask = torch.zeros(self.n_domains, self.n_intents, dtype=torch.bool)
        for d_lbl, i_lbl in zip(y_domain, y_intent):
            mask[d_idx[d_lbl], i_idx[i_lbl]] = True
        self.register_buffer("domain_intent_mask", mask)
        self.train()

    def _prepare_dataset(
        self,
        X: list[str],
        y_domain: list[str],
        y_intent: list[str],
        max_length: int = 512,
    ) -> HierarchicalTextDataset:
        X = [x[: max_length * 10] for x in X]
        tokenized = [
            enc.ids[:max_length]
            for enc in self.tokenizer.encode_batch_fast(
                X, add_special_tokens=False
            )
        ]
        d_idx = {d: i for i, d in enumerate(self.domain_classes_)}
        i_idx = {i: j for j, i in enumerate(self.intent_classes_)}
        d_tensor = torch.tensor([d_idx[d] for d in y_domain], dtype=torch.long)
        i_tensor = torch.tensor([i_idx[i] for i in y_intent], dtype=torch.long)
        return HierarchicalTextDataset(tokenized, d_tensor, i_tensor)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "minishlab/potion-base-32M",
        n_layers: int = N_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        freeze: bool = FREEZE_EMBEDDINGS,
        **kwargs,
    ) -> "StaticModelForHierarchicalClassification":
        """Initialise from a HuggingFace model name or local path."""
        static = StaticModel.from_pretrained(model_name)
        return cls.from_static_model(
            static, n_layers=n_layers, hidden_dim=hidden_dim, freeze=freeze,
            **kwargs,
        )

    @classmethod
    def from_static_model(
        cls,
        model: StaticModel,
        n_layers: int = N_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        freeze: bool = FREEZE_EMBEDDINGS,
        pad_token: str = "[PAD]",
        **kwargs,
    ) -> "StaticModelForHierarchicalClassification":
        """Wrap an already-loaded StaticModel."""
        model.embedding = np.nan_to_num(model.embedding)
        vectors = torch.from_numpy(model.embedding)
        weights = (
            torch.from_numpy(model.weights) if model.weights is not None else None
        )
        token_mapping = (
            model.token_mapping.tolist()
            if model.token_mapping is not None
            else None
        )
        return cls(
            vectors=vectors,
            tokenizer=model.tokenizer,
            n_domains=2,  # placeholder — overwritten by fit()
            n_intents=2,  # placeholder — overwritten by fit()
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            pad_id=model.tokenizer.token_to_id(pad_token),
            token_mapping=token_mapping,
            weights=weights,
            freeze=freeze,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
    ) -> "StaticModelForHierarchicalClassification":
        """Reload a saved model from a ``.pt`` checkpoint produced by training."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        static = StaticModel.from_pretrained(ckpt["base_model"])
        instance = cls.from_static_model(
            static,
            n_layers=ckpt["n_layers"],
            hidden_dim=ckpt["hidden_dim"],
        )
        instance._initialize_classes(ckpt["domain_classes"], ckpt["intent_classes"])
        instance.load_state_dict(ckpt["state_dict"])
        instance.eval()
        if device != "cpu":
            instance.to(device)
        return instance


# ---------------------------------------------------------------------------
# Lightning training module (imported only during fit())
# ---------------------------------------------------------------------------

class _HierarchicalLightningModule:
    """Defined here so the class can be instantiated by fit() without a
    top-level lightning import.  Actual subclassing happens lazily."""

    def __new__(cls, model, learning_rate, lambda_intent=1.0):
        try:
            import lightning as pl
            from lightning.pytorch.utilities.types import OptimizerLRScheduler
        except ImportError:
            raise ImportError(
                "Training requires 'lightning'. "
                "pip install ovos-media-classifier[train]"
            )

        class _Module(pl.LightningModule):
            def __init__(self, model, learning_rate, lambda_intent=1.0):
                super().__init__()
                self.model = model
                self.learning_rate = learning_rate
                self.lambda_intent = lambda_intent
                self.domain_loss_fn = nn.CrossEntropyLoss()
                self.intent_loss_fn = nn.CrossEntropyLoss()

            def _step(self, batch):
                x, y_domain, y_intent = batch
                (d_logits, i_logits), _ = self.model(x)
                loss = (
                    self.domain_loss_fn(d_logits, y_domain)
                    + self.lambda_intent * self.intent_loss_fn(i_logits, y_intent)
                )
                d_acc = (d_logits.argmax(1) == y_domain).float().mean()
                i_acc = (i_logits.argmax(1) == y_intent).float().mean()
                return loss, d_acc, i_acc

            def training_step(self, batch, batch_idx):
                loss, d_acc, i_acc = self._step(batch)
                self.log("train_loss", loss, prog_bar=True)
                self.log("train_domain_acc", d_acc)
                self.log("train_intent_acc", i_acc)
                return loss

            def validation_step(self, batch, batch_idx):
                loss, d_acc, i_acc = self._step(batch)
                self.log("val_loss", loss, prog_bar=True)
                self.log("val_domain_acc", d_acc, prog_bar=True)
                self.log("val_intent_acc", i_acc, prog_bar=True)
                return loss

            def configure_optimizers(self):
                opt = torch.optim.Adam(
                    self.model.parameters(), lr=self.learning_rate
                )
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    threshold=0.03,
                )
                return {
                    "optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
                }

        return _Module(model, learning_rate, lambda_intent)
