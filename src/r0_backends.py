"""
Multi-pLM / substitution R0 backends for TreeSBM reference mutation priors.

Frozen R0 mutation prior backends:
  JTT/WAG/LG substitution | ESM-2-650M ± fitness | ESM-C ± fitness | ProGen2 | …
TreeSBM keeps ``log R_θ = log R0(+tilt) + c_θ``; these wrappers only swap how
``log R0`` is produced. They are adapters, not new networks.

Fitness tilting stays in ``src/bridge/fitness_tilt.py`` (orthogonal to backend).
Poisson branching λ(x) stays in ``src/reference_process.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

# Canonical names used by CLI / cache tags / paper rows.
BACKEND_ESM2 = "esm2"
BACKEND_ESM2_650M = "esm2_650m"
BACKEND_ESMC = "esmc"
BACKEND_PROGEN2 = "progen2"
BACKEND_EVO2 = "evo2"
BACKEND_JTT = "jtt"
BACKEND_WAG = "wag"
BACKEND_LG = "lg"
BACKEND_NEUTRAL = "neutral"

SUBSTITUTION_BACKENDS = frozenset(
    {BACKEND_JTT, BACKEND_WAG, BACKEND_LG, BACKEND_NEUTRAL}
)
STUB_BACKENDS = frozenset({BACKEND_PROGEN2, BACKEND_EVO2})

DEFAULT_ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_ESM2_650M_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESMC_MODEL = "esmc_300m"

# Cache file tag → ``group_{g:03d}_ref_rates{tag}.pt``
# Empty tag preserves legacy ``group_*_ref_rates.pt`` (current ESM-2-8M caches).
BACKEND_CACHE_TAG = {
    BACKEND_ESM2: "",
    BACKEND_ESM2_650M: "_esm2_650m",
    BACKEND_ESMC: "_esmc",
    BACKEND_PROGEN2: "_progen2",
    BACKEND_EVO2: "_evo2",
    BACKEND_JTT: "_jtt",
    BACKEND_WAG: "_wag",
    BACKEND_LG: "_lg",
    BACKEND_NEUTRAL: "_neutral",
}


def normalize_backend_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "esm2_8m": BACKEND_ESM2,
        "esm_2": BACKEND_ESM2,
        "esm2_t6": BACKEND_ESM2,
        "esm2_650": BACKEND_ESM2_650M,
        "esm2_t33": BACKEND_ESM2_650M,
        "esm_c": BACKEND_ESMC,
        "esm_cambrian": BACKEND_ESMC,
        "progen": BACKEND_PROGEN2,
        "progen_2": BACKEND_PROGEN2,
        "evo_2": BACKEND_EVO2,
        "substitution": BACKEND_JTT,
        "substitution_only": BACKEND_JTT,
    }
    return aliases.get(key, key)


def cache_tag_for_backend(backend: str, override: Optional[str] = None) -> str:
    """Return filename tag ('' or '_esmc', …). ``override`` wins if not None."""
    if override is not None:
        tag = override.strip()
        if tag and not tag.startswith("_"):
            tag = "_" + tag
        return tag
    name = normalize_backend_name(backend)
    if name not in BACKEND_CACHE_TAG:
        raise ValueError(
            f"Unknown R0 backend {backend!r}. "
            f"Known: {sorted(BACKEND_CACHE_TAG)}"
        )
    return BACKEND_CACHE_TAG[name]


def ref_rates_filename(group: int, tag: str = "") -> str:
    return f"group_{group:03d}_ref_rates{tag}.pt"


class R0Backend(ABC):
    """Frozen mutation prior → sitewise log-probs over 20 AAs."""

    name: str = "base"

    @abstractmethod
    def log_mutation_rates(
        self,
        sequences: Sequence[str],
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequences: AA strings (may contain gaps '-').
            max_seq_len: truncate / pad length L.
            device: optional torch device for compute; output may stay on CPU.

        Returns:
            log_R0: float tensor [N, L, 20] (typically log-softmax over AA).
        """

    def close(self) -> None:
        """Optional resource cleanup."""
        return None


class StubR0Backend(R0Backend):
    """Explicit stub for heavy / unavailable priors (ProGen2, Evo2)."""

    def __init__(self, name: str, install_hint: str):
        self.name = name
        self.install_hint = install_hint

    def log_mutation_rates(self, sequences, max_seq_len, device=None):
        raise NotImplementedError(
            f"R0 backend {self.name!r} is stubbed (too heavy / not wired). "
            f"{self.install_hint}"
        )


class ESM2R0Backend(R0Backend):
    """HuggingFace ESM-2 masked LM → per-position AA log-probs (legacy default)."""

    def __init__(
        self,
        model_id: str = DEFAULT_ESM2_MODEL,
        device: Optional[str] = None,
        name: str = BACKEND_ESM2,
    ):
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.name = name
        self.model_id = model_id
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = EsmForMaskedLM.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.aa_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_VOCAB],
            dtype=torch.long,
            device=self.device,
        )

    @torch.no_grad()
    def log_mutation_rates(self, sequences, max_seq_len, device=None):
        dev = torch.device(device) if device is not None else self.device
        if dev != self.device:
            # Move model if caller asks for a different device (rare).
            self.model.to(dev)
            self.aa_token_ids = self.aa_token_ids.to(dev)
            self.device = dev

        N, L = len(sequences), max_seq_len
        log_rates = torch.zeros(N, L, 20, dtype=torch.float32)
        # Batch in one shot when N is small; callers may chunk externally.
        tokens = self.tokenizer(
            list(sequences), return_tensors="pt", padding=True, truncation=False
        ).to(self.device)
        logits = self.model(**tokens).logits
        seq_lens = tokens["attention_mask"].sum(dim=1)
        for i in range(N):
            actual_L = int(seq_lens[i].item()) - 2  # strip BOS/EOS
            if actual_L <= 0:
                continue
            aa_logits = logits[i, 1 : actual_L + 1, :][:, self.aa_token_ids]
            log_probs = F.log_softmax(aa_logits, dim=-1)
            clip = min(actual_L, L)
            log_rates[i, :clip, :] = log_probs[:clip].cpu()
        return log_rates


class ESMCR0Backend(R0Backend):
    """
    ESM-C (Cambrian) via EvolutionaryScale ``esm`` package when installed.

    Preferred local open weights: ``esmc_300m`` (lightweight) or ``esmc_600m``.
    Falls back to a clear ImportError / runtime message if ``esm`` is missing.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_ESMC_MODEL,
        device: Optional[str] = None,
    ):
        self.name = BACKEND_ESMC
        self.model_id = model_id
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as e:
            raise ImportError(
                "ESM-C backend requires the EvolutionaryScale `esm` package. "
                "Install with: pip install esm  "
                "(or pip install 'esm @ git+https://github.com/evolutionaryscale/esm.git'). "
                "Then retry --r0-backend esmc."
            ) from e

        self._ESMProtein = ESMProtein
        self._LogitsConfig = LogitsConfig
        self.model = ESMC.from_pretrained(model_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Map ESM-C vocab token ids for the 20 standard AAs (built once).
        self._aa_index = self._build_aa_index()

    def _build_aa_index(self) -> torch.Tensor:
        """Resolve token ids for ACDEFGHIKLMNPQRSTVWY in the ESM-C tokenizer."""
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            # Fallback: assume contiguous AA block like ESM-2 (indices 4..23).
            return torch.arange(4, 24, dtype=torch.long, device=self.device)
        ids = []
        for aa in AA_VOCAB:
            # Prefer single-letter encode helpers when present.
            if hasattr(tok, "encode"):
                try:
                    enc = tok.encode(aa)
                    if isinstance(enc, (list, tuple)) and len(enc) >= 1:
                        ids.append(int(enc[0] if not hasattr(enc[0], "item") else enc[0]))
                        continue
                except Exception:
                    pass
            if hasattr(tok, "get_vocab"):
                vocab = tok.get_vocab()
                if aa in vocab:
                    ids.append(int(vocab[aa]))
                    continue
            raise RuntimeError(
                f"Could not map amino acid {aa!r} in ESM-C tokenizer; "
                "please upgrade `esm` or use --r0-backend esm2."
            )
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def log_mutation_rates(self, sequences, max_seq_len, device=None):
        N, L = len(sequences), max_seq_len
        log_rates = torch.zeros(N, L, 20, dtype=torch.float32)
        cfg = self._LogitsConfig(sequence=True, return_embeddings=False)
        for i, seq in enumerate(sequences):
            clean = seq.replace("-", "X")[:L]
            if not clean:
                continue
            protein = self._ESMProtein(sequence=clean)
            protein_tensor = self.model.encode(protein)
            out = self.model.logits(protein_tensor, cfg)
            logits = out.logits
            # logits may be [1, L_tok, V] or an object with .sequence
            if hasattr(logits, "sequence"):
                logits = logits.sequence
            if not torch.is_tensor(logits):
                raise RuntimeError(
                    f"Unexpected ESM-C logits type: {type(logits)}; "
                    "expected a tensor with AA vocabulary dims."
                )
            if logits.dim() == 3:
                logits = logits[0]
            # Drop BOS/EOS if present (length = len(seq)+2).
            if logits.size(0) == len(clean) + 2:
                logits = logits[1:-1]
            elif logits.size(0) > len(clean):
                logits = logits[: len(clean)]
            aa_logits = logits[: len(clean), :][:, self._aa_index.to(logits.device)]
            log_probs = F.log_softmax(aa_logits.float(), dim=-1)
            clip = min(log_probs.size(0), L)
            log_rates[i, :clip, :] = log_probs[:clip].cpu()
        return log_rates


class SubstitutionMatrixR0Backend(R0Backend):
    """
    Sitewise destination log-probs from an AA substitution rate matrix (JTT/WAG/LG).

    For current residue a at each site, build a destination distribution from the
    CTMC jump chain of Q, with residual stay mass on a. Context-independent —
    empirical substitution-matrix R0 (JTT / WAG / LG).
    """

    def __init__(self, model: str = "JTT", stay_mass: float = 0.5):
        self.model_name = model.upper() if model.lower() != "neutral" else "NEUTRAL"
        self.name = self.model_name.lower()
        self.stay_mass = float(stay_mass)
        self._Q = self._load_rate_matrix(self.model_name)  # [20, 20] numpy

    @staticmethod
    def _load_rate_matrix(model_name: str):
        import numpy as np

        if model_name == "NEUTRAL":
            Q = np.full((20, 20), 1.0, dtype=np.float64)
            np.fill_diagonal(Q, -19.0)
            return Q
        try:
            import pyvolve
        except ImportError as e:
            raise ImportError(
                f"Substitution backend {model_name} needs pyvolve "
                "(see requirements.txt)."
            ) from e
        m = pyvolve.Model(model_name)
        # pyvolve stores instantaneous rate matrix as .matrix or similar.
        Q = getattr(m, "matrix", None)
        if Q is None and hasattr(m, "params"):
            Q = m.params.get("matrix")
        if Q is None:
            raise RuntimeError(
                f"Could not extract rate matrix from pyvolve.Model({model_name!r})"
            )
        Q = np.asarray(Q, dtype=np.float64)
        if Q.shape != (20, 20):
            raise RuntimeError(f"Expected 20x20 rate matrix, got {Q.shape}")
        return Q

    def log_mutation_rates(self, sequences, max_seq_len, device=None):
        import numpy as np

        N, L = len(sequences), max_seq_len
        log_rates = torch.zeros(N, L, 20, dtype=torch.float32)
        stay = min(max(self.stay_mass, 1e-6), 1.0 - 1e-6)
        Q = self._Q
        for i, seq in enumerate(sequences):
            for pos, aa in enumerate(seq[:L]):
                a = AA_TO_IDX.get(aa)
                if a is None:
                    # gap / unknown → uniform
                    log_rates[i, pos, :] = -np.log(20.0)
                    continue
                row = Q[a].copy()
                off = np.maximum(row, 0.0)
                off[a] = 0.0
                s = off.sum()
                probs = np.full(20, (1.0 - stay) / 19.0, dtype=np.float64)
                if s > 1e-12:
                    probs = off / s * (1.0 - stay)
                probs[a] = stay
                probs = np.clip(probs, 1e-12, None)
                probs /= probs.sum()
                log_rates[i, pos, :] = torch.from_numpy(np.log(probs)).float()
        return log_rates


def build_r0_backend(
    backend: str,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> R0Backend:
    """
    Factory for frozen R0 backends.

    ``model_id`` overrides the default HF / ESM-C checkpoint when applicable.
    """
    name = normalize_backend_name(backend)
    if name in STUB_BACKENDS:
        hints = {
            BACKEND_PROGEN2: (
                "ProGen2 is not bundled; install salesforce/progen and wire a "
                "causal-LM → site marginal adapter, or skip this backend."
            ),
            BACKEND_EVO2: (
                "Evo2 is heavy (OOM risk on mig GPUs); use API/distill or skip "
                "the appendix D.1 Evo2 row for now."
            ),
        }
        return StubR0Backend(name, hints[name])

    if name == BACKEND_ESM2:
        return ESM2R0Backend(
            model_id=model_id or DEFAULT_ESM2_MODEL,
            device=device,
            name=BACKEND_ESM2,
        )
    if name == BACKEND_ESM2_650M:
        return ESM2R0Backend(
            model_id=model_id or DEFAULT_ESM2_650M_MODEL,
            device=device,
            name=BACKEND_ESM2_650M,
        )
    if name == BACKEND_ESMC:
        return ESMCR0Backend(model_id=model_id or DEFAULT_ESMC_MODEL, device=device)
    if name in SUBSTITUTION_BACKENDS:
        return SubstitutionMatrixR0Backend(model=name)

    raise ValueError(
        f"Unknown R0 backend {backend!r}. "
        f"Choose from: esm2, esm2_650m, esmc, progen2, evo2, jtt, wag, lg, neutral."
    )


def list_backends() -> list[str]:
    return sorted(BACKEND_CACHE_TAG.keys())
