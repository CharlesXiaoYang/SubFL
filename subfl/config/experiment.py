"""Configuration of a run; the whole ExperimentConfig is saved as manifest.json."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset and client-split options."""

    dataset_name: str = "cifar10"
    root: Path = Path("./data/cifar10")
    num_clients: int = 10
    batch_size: int = 256
    num_workers: int = 8
    image_size: int = 32  # recorded only; every image is 32x32
    # "cifar10" (fixed constants) | "public_pool" (SVHN only: per-channel mean and std of the public pool P)
    normalization: str = "cifar10"
    public_samples_per_class: int = 0
    private_split_seed: int = 1234
    dirichlet_alpha: float = 100.0
    # "iid" | "dirichlet", set by train.py ("auto" is only the value pretrain.py records)
    partition: str = "auto"
    # download a missing dataset into root (train.py: only with --download_data)
    download: bool = True
    # "none" | "public_pool" (the public pool P, evaluated like the test set)
    val_source: str = "none"
    auxiliary_dataset_name: str = "matched"
    auxiliary_root: Path = Path("./data/auxiliary")
    auxiliary_random_labels: bool = True
    svhn_train_split: str = "train"
    # source of SVHN's public pool P: "same" (svhn_train_split, the rest private) | "extra" (all of 'train' private)
    svhn_public_split: str = "same"


@dataclass
class ModelConfig:
    """Model construction options."""

    model_name: str = "cifar_resnet20"
    num_classes: int = 10
    pretrained_weights: str = "none"
    freeze_backbone: bool = False
    trainable_scope: str = "full"
    head_type: str = "linear"
    head_hidden_dim: int = 256
    head_dropout: float = 0.0
    init_checkpoint: str = ""  # warm-start checkpoint; "" = random init
    allow_partial_init: bool = False  # recorded only
    init_checkpoint_sha256: str = ""  # required sha256 prefix of init_checkpoint
    checkpoint_path: str = ""


@dataclass
class TrainConfig:
    """Optimization settings of local client training."""

    num_rounds: int = 200
    local_epochs: int = 8
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 1234
    device: str = "auto"
    client_parallelism: int = 1
    model_dtype: str = "fp32"
    update_dtype: str = "match_model"
    server_momentum: float = 0.0
    augmult_k: int = 1  # augmented views per example, averaged before clipping; 1 = off
    ema_decay: float = 0.0  # EMA of the global weights, evaluation only; 0 = off
    ema_decay2: float = 0.0  # a second EMA decay; 0 = off
    lr_schedule: str = "constant"  # learning-rate schedule over rounds: constant | cosine
    lr_min_ratio: float = 0.0  # final lr as a fraction of lr for cosine
    diagnostics: str = "off"  # "off": no statistics of private data beyond the DP release


@dataclass
class PrivacyConfig:
    """Record-level DP parameters."""

    total_epsilon: float = 10.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    adaptive_schedule: str = "none"
    accountant: str = "basic"
    client_sample_rate: float = 1.0  # every client in every round
    rdp_orders: str = "2-64"
    noise_multiplier: float = 0.0
    dense_noise_mode: str = "dpsgd"  # per-example clip and Gaussian noise at every local step
    lot_divisor: str = "expected"  # divisor of each noisy lot sum: the constant min(B, n_c)
    sampler: str = "poisson"  # each record joins a local lot with probability q (what the RDP accountant assumes)
    gep_accounting: str = "joint"  # a GEP step (embedding + residual) is one Gaussian mechanism
    dense_z_calibration: str = "per_client"  # each client calibrates its noise multiplier from its own n_c


@dataclass
class GEPConfig:
    """Subspace (GEP) options; fields the code does not read keep the manifest layout of the archived runs."""

    enabled: bool = False
    anchor_samples: int = 256
    subspace_dim: int = 64
    power_iterations: int = 1
    recompute_interval: int = 1
    local_recompute_interval: int = 1  # refit the subspace every N local steps
    freeze_anchor_after_round: int = -1
    warmup_rounds: int = 0
    transition_rounds: int = 0
    history_size: int = 20
    history_decay: float = 1.0
    embedding_clip_norm: float = 1.0
    residual_clip_norm: float = 0.2
    upload_residual: bool = True
    # "anchor_svd" (top right singular vectors of the public anchor gradients) | "random" (a random subspace of
    # the same per-tensor ranks)
    subspace_source: str = "anchor_svd"
    embedding_epsilon_ratio: float = 0.5  # share of the privacy budget given to the embedding; 0.5 = equal split
    epsilon_split_strategy: str = "fixed"
    min_embedding_epsilon_ratio: float = 0.3
    max_embedding_epsilon_ratio: float = 0.8
    aux_source: str = "matched"
    anchor_source: str = "hybrid"
    anchor_auxiliary_mode: str = "gradients"
    anchor_update_count: int = 64
    anchor_update_batch_size: int = 32
    anchor_update_steps: int = 1
    anchor_update_lr: float = 0.03
    anchor_update_momentum: float = 0.0
    aux_weight: float = 1.0
    history_weight: float = 1.0
    group_by: str = "top_level"
    rank_allocation: str = "energy"  # split the rank across tensors by their anchor gradient energy
    dense_head: bool = False
    dense_head_group_name: str = "fc_out"
    dense_head_clip_norm: float = 1.0
    persistent_dense_alpha: float = 0.0
    dictionary_window_rounds: int = 10
    basis_source: str = "released_dictionary"
    dictionary_update_policy: str = "dense_only"
    refresh_interval: int = 5
    explore_interval: int = 20
    freeze_basis_after_warmup: bool = False
    public_simfl_clients: int = 10
    public_simfl_local_epochs: int = 0
    public_simfl_batch_size: int = 0
    public_simfl_samples_per_client: int = 0
    public_simfl_restarts: int = 1
    public_simfl_history_refreshes: int = 1
    public_simfl_history_decay: float = 1.0
    protect_classifier_head: bool = True
    classifier_head_names: str = "fc_out,classifier"
    full_rank_bias: bool = True
    head_full_rank_threshold: int = 8192
    min_subspace_rank: int = 1
    method_variant: str = "pfa"
    schedule_kind: str = "random_capped"
    per_client_high_rounds_K: int = 2
    high_clients_per_round: int = 0
    normal_participation_q: float = 0.5
    high_budget_fraction_f: float = 0.5
    basis_window_rounds: int = 30
    basis_decay: float = 1.0
    min_rows_for_basis: int = 10
    drop_residual_on_normal: bool = True
    bootstrap_mode: str = "high_only_until_basis"
    aggregation_weighting: str = "balanced"
    public_client_fraction: float = 0.5
    public_client_seed: int = 1234
    public_participation_q: float = 1.0
    private_participation_q: float = 1.0
    public_role_mode: str = "fixed"
    public_clients_per_round: int = 0
    private_upload_mode: str = "gep"
    private_residual_scope: str = "all"
    private_residual_patterns: str = ""
    basis_personalization: str = "none"
    global_subspace_dim: int = 0
    client_subspace_dim: int = 128
    client_basis_window_rounds: int = 30
    client_basis_decay: float = 1.0
    min_client_basis_rows: int = 3
    client_basis_refresh_interval: int = 5
    pfa_oracle_diagnostics: bool = False
    pfa_private_oracle_diagnostics: bool = False
    pfa_private_oracle_interval: int = 5
    pfa_private_oracle_holdout_fraction: float = 0.5
    pfa_private_oracle_basis_dim: int = 0
    pfa_verbose_clients: bool = False
    pfa_dictionary_dtype: str = "fp16"
    temporal_warmup_rounds: int = 20
    temporal_private_per_clean: int = 1
    temporal_buffer_window: int = 10


@dataclass
class LayerwiseConfig:
    """Not read by the code; kept for the manifest layout."""

    enabled: bool = False
    group_by: str = "top_level"
    groups_per_round: int = 1
    selection_strategy: str = "uniform_random"
    epsilon_allocation: str = "size"
    delta_allocation: str = "uniform"
    freeze_unselected: bool = True
    sparse_aggregation: bool = True


@dataclass
class ScaffoldConfig:
    """Not read by the code; kept for the manifest layout."""

    enabled: bool = False
    control_clip_norm: float = 5.0
    warmstart_rounds: int = 4


@dataclass
class OutputConfig:
    """Output location and which files a run saves."""

    output_root: Path = Path("runs")
    save_plots: bool = True  # recorded only; no plots are written
    save_metrics_csv: bool = True
    save_manifest_json: bool = True
    save_model_every: int = 0  # save the global model every N rounds and after the last round; 0 = never


@dataclass
class ExperimentConfig:
    """Single object that captures a fully resolved experiment setup."""

    experiment_name: str
    method: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    gep: GEPConfig = field(default_factory=GEPConfig)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    layerwise: LayerwiseConfig = field(default_factory=LayerwiseConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
