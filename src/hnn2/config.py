"""Scientific defaults and the coordinates identifying each experiment run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import warnings

# The model variants (SPEC.md, sections 1 and 2):
#   rec    -- I driven by E output (feedback about network activity)
#   ff     -- I driven by the input (tracks stimulus, not E's response)
#   thresh -- no I unit; the excitatory threshold adapts instead of inhibition
MODEL_CODES = ("rec", "ff", "thresh")

SCHEMA_VERSION = 1   # bumped when the shape of a saved config changes


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed scientific parameters, passed explicitly by the experiment script."""

    schema_version: int = SCHEMA_VERSION

    # Dataset recipe; prepared explicitly and reused from a saved NPZ.
    image_size: tuple[int, int] = (10, 10)          # 100 inputs; 28x28 is too slow to settle
    raw_baseline_size: tuple[int, int] = (22, 22)   # 484 = n_excitatory: equal-size baseline
    n_samples_per_split: int = 1024               # class-stratified across all ten digits
    train_val_ratio: float = 0.9
    data_seed: int = 0

    # One I neuron supplies shared inhibition through per-unit W_EI weights.
    # exp(-1/tau) retains about 0.61 of E and 0.95 of I state per update
    # at these defaults (SPEC.md, section 1).
    n_excitatory: int = 484
    n_inhibitory: int = 1
    tau_e: float = 2.0
    tau_i: float = 20.0
    theta_init: float = 0.145        # neurons output max(v_E - theta, 0)

    # Iterate the map for a fixed number of updates to approximate its steady
    # response. That E output is the representation later analysis consumes.
    settle_steps: int = 200          # V1: number of state updates
    n_stabilise: int = 10            # C5: warm-start sample count (train split — V3)

    # One homeostatic update per settled batch; eta follows SPEC.md, equation (8).
    n_epochs: int = 10
    batch_size: int = 64
    eta_decay: float = 0.95
    eta_decay_start_epoch: int = 3

    # Weight on Var_i in the monitoring loss (mean - targ)^2 + w*Var_i. At 1.0 the
    # loss is exactly the per-unit MSE mean_i (r_i - targ)^2 (SPEC V10), the
    # objective the learning-rate selection minimises (hp.select). The learning
    # rule itself never reads it.
    monitor_var_weight: float = 1.0

    # Adaptation probes (SPEC V6); the classifier is evaluated separately.
    monitor_probe_split: str = "val"      # every epoch
    analysis_probe_split: str = "test"    # after the first and final epochs

    # A linear classifier fitted on frozen activity. Features stay raw, since z-scoring
    # would erase the firing-rate differences being measured.
    readout_epochs: int = 15
    readout_lr: float = 3e-3
    readout_eval_every: int = 1       # validate every N steps; 1 = every batch (D11)
    readout_standardize: bool = False
    # Also evaluate the final classifier on the full test split (10k for MNIST).
    readout_final_eval_full_test: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.train_val_ratio < 1.0):
            raise ValueError(f"train_val_ratio must be in (0, 1), got {self.train_val_ratio}")
        if self.settle_steps < 1:
            raise ValueError("settle_steps must be >= 1")
        if self.monitor_probe_split not in ("train", "val", "test"):
            raise ValueError(f"invalid monitor_probe_split: {self.monitor_probe_split}")
        if self.analysis_probe_split not in ("train", "val", "test"):
            raise ValueError(f"invalid analysis_probe_split: {self.analysis_probe_split}")


@dataclass(frozen=True)
class RunSpec:
    """The four coordinates identifying one run.

    Kept separate from the shared scientific parameters for explicit condition loops.
    ``seed_index`` indexes ``rng.PRIMES``; it is not a raw seed value.
    """

    model_code: str
    targ: float
    eta: float
    seed_index: int

    def __post_init__(self) -> None:
        if self.model_code not in MODEL_CODES:
            raise ValueError(f"model_code must be one of {MODEL_CODES}, got {self.model_code!r}")
        if self.targ <= 0:
            raise ValueError(f"targ must be positive, got {self.targ}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if self.seed_index < 0:
            raise ValueError(f"seed_index must be >= 0, got {self.seed_index}")

    def in_original_regime(self, config: ExperimentConfig) -> bool:
        """Return whether ``targ > theta_init``, the original study's search regime.

        Runs outside it trigger an informational warning in ``validate_against``
        but are still allowed (SPEC V7).
        """
        return self.targ > config.theta_init

    def validate_against(self, config: ExperimentConfig) -> None:
        """Warn, never raise, when ``targ <= theta_init`` (design V7)."""
        if not self.in_original_regime(config):
            warnings.warn(
                f"targ={self.targ} <= theta_init={config.theta_init}: outside the regime "
                "explored by the original study (informational only — not an error).",
                stacklevel=2,
            )


def config_to_json(config: ExperimentConfig) -> str:
    """Serialise scientific conditions with stable key order and finite JSON values."""
    return json.dumps(asdict(config), sort_keys=True, allow_nan=False)


def experiment_config_from_json(text: str) -> ExperimentConfig:
    """Restore scientific conditions, including the tuple-valued image dimensions."""
    values = json.loads(text)
    for key in ("image_size", "raw_baseline_size"):
        if key in values:
            values[key] = tuple(values[key])
    return ExperimentConfig(**values)
