from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    path: str
    image_size: int
    multires: bool
    batch_size: int
    max_workers: int
    latent_field: str
    caption_field: str
    alt_caption_field: str
    alt_caption_min_length: int
    alt_caption_probability: float
    scale_factor: int
    image_field: str | None
    type: str = "huggingface_disk"


@dataclass
class ModelConfig:
    channels: int
    n_layers: int
    n_encoder_layers: int
    n_heads: int
    dim: int
    patch_size: int
    use_tread: bool
    compile: bool = False
    type: str = "ImageModel"
    # ImageModel specific parameters
    decoder_hidden_size: int = 64
    num_text_blocks: int = 4
    # DDT specific parameters
    experiment: str = "baseline"
    num_classes: int = 1000
    learn_sigma: bool = True
    deep_supervision: int = 0
    use_cross_attention: bool = False
    tread_dropout_ratio: float = 0.5
    tread_prev_blocks: int = 3
    tread_post_blocks: int = 1


@dataclass
class VAEConfig:
    path: str
    class_name: str = "AutoencoderKLVAE"

@dataclass
class TextEncoderConfig:
    path: str | None = None
    max_length: int = 0
    txt_embed_dim: int = 0
    type: str = "causal_lm"
    default_label: int = 0


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    warmup_steps: int


@dataclass
class TrainingSchedule:
    epochs: int
    mixed_precision: str
    resume: bool
    save_every: int
    cond_dropout: float
    clip_grad: float
    log_window: int
    unconditional_caption: str


@dataclass
class SamplingConfig:
    interval: int
    steps: int
    num_samples: int
    cfg: float


@dataclass
class LoggingConfig:
    checkpoints_dir: str
    samples_dir: str
    runs_dir: str
    track_performance: bool = False


@dataclass
class LoopConfig:
    train_step: str = "default"
    sample_step: str = "default"

@dataclass
class TrainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    vae: VAEConfig
    text_encoder: TextEncoderConfig
    optimizer: OptimizerConfig
    training: TrainingSchedule
    sampling: SamplingConfig
    logging: LoggingConfig
    loop: LoopConfig
    losses: dict[str, float]


@dataclass
class InferenceParams:
    caption: str
    cfg: float
    steps: int
    height: int
    width: int
    seed: int
    negative_caption: str


@dataclass
class InferenceIO:
    checkpoint_dir: str
    output_path: str


@dataclass
class InferenceConfig:
    model: ModelConfig
    vae: VAEConfig
    text_encoder: TextEncoderConfig
    inference: InferenceParams
    io: InferenceIO


@dataclass
class CacheShardConfig:
    directory: str
    pattern: str
    min_index: int
    max_index: int


@dataclass
class CacheProcessingConfig:
    bucket_size: int
    save_threshold: int
    min_caption_length: int
    multires: bool = True


@dataclass
class CacheTextLookupConfig:
    type: str | None = None  # e.g., "shelve", "sqlite", "jsonl"
    path: str | None = None

@dataclass
class CacheOutputConfig:
    cache_dir: str


@dataclass
class CacheDataloaderConfig:
    max_workers: int | None = None


@dataclass
class CacheFlowersDatasetConfig:
    path: str
    split: str = "train"
    image_field: str = "image"
    label_field: str = "label"


@dataclass
class CacheImageFolderConfig:
    root: str
    extensions: list[str] | None = None
    recursive: bool = False


@dataclass
class CacheHFParquetDatasetConfig:
    data_files: str | list[str]
    image_field: str = "image"
    label_field: str = "label"


@dataclass
class CacheDenseFusionDatasetConfig:
    jsonl_path: str
    image_root: str | None = None
    image_field: str = "image_path"
    caption_field: str = "caption"
    id_field: str = "image_id"


@dataclass
class CacheDatasetConfig:
    type: str = "webdataset"
    webdataset: CacheShardConfig | None = None
    flowers: CacheFlowersDatasetConfig | None = None
    image_folder: CacheImageFolderConfig | None = None
    hf_parquet: CacheHFParquetDatasetConfig | None = None
    densefusion: CacheDenseFusionDatasetConfig | None = None


@dataclass
class CacheConfig:
    dataset: CacheDatasetConfig
    processing: CacheProcessingConfig
    output: CacheOutputConfig
    vae: VAEConfig
    caption_text_lookup: CacheTextLookupConfig = field(default_factory=CacheTextLookupConfig)
    alt_text_lookup: CacheTextLookupConfig = field(default_factory=CacheTextLookupConfig)
    dataloader: CacheDataloaderConfig = field(default_factory=CacheDataloaderConfig)
    aspect_ratio_bucket: int = 256


def load_train_config(path: str) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text())
    loop_data = data.get("loop") or {}
    losses_data = data.get("losses")
    if not losses_data:
        raise ValueError("Train config must define a 'losses' mapping with weights.")
    optimizer_data = dict(data["optimizer"])
    optimizer_data.setdefault("name", "adam")

    return TrainConfig(
        dataset=DatasetConfig(**data["dataset"]),
        model=ModelConfig(**data["model"]),
        vae=VAEConfig(**data["vae"]),
        text_encoder=TextEncoderConfig(**data["text_encoder"]),
        optimizer=OptimizerConfig(**optimizer_data),
        training=TrainingSchedule(**data["training"]),
        sampling=SamplingConfig(**data["sampling"]),
        logging=LoggingConfig(**data["logging"]),
        loop=LoopConfig(**loop_data),
        losses={str(k): float(v) for k, v in losses_data.items()},
    )


def load_inference_config(path: str) -> InferenceConfig:
    data = yaml.safe_load(Path(path).read_text())
    return InferenceConfig(
        model=ModelConfig(**data["model"]),
        vae=VAEConfig(**data["vae"]),
        text_encoder=TextEncoderConfig(**data["text_encoder"]),
        inference=InferenceParams(**data["inference"]),
        io=InferenceIO(**data["io"]),
    )


def load_cache_config(path: str) -> CacheConfig:
    data = yaml.safe_load(Path(path).read_text())
    dataloader_data = data.get("dataloader") or {}
    # New fields for separate caption/alt lookups; default to empty
    caption_lookup_data = data.get("caption_text_lookup") or {}
    alt_lookup_data = data.get("alt_text_lookup") or {}
    dataset_data = data.get("dataset")
    if dataset_data is None:
        shards_data = data.get("shards")
        if shards_data is None:
            raise ValueError("Cache config must define a 'dataset' section or legacy 'shards'.")
        dataset_data = {"type": "webdataset", "webdataset": shards_data}

    dataset_type = dataset_data.get("type", "webdataset")
    webdataset_data = dataset_data.get("webdataset")
    flowers_data = dataset_data.get("flowers")
    image_folder_data = dataset_data.get("image_folder")
    hf_parquet_data = dataset_data.get("hf_parquet")
    densefusion_data = dataset_data.get("densefusion")

    dataset_cfg = CacheDatasetConfig(
        type=str(dataset_type),
        webdataset=CacheShardConfig(**webdataset_data) if webdataset_data else None,
        flowers=CacheFlowersDatasetConfig(**flowers_data) if flowers_data else None,
        image_folder=CacheImageFolderConfig(**image_folder_data) if image_folder_data else None,
        hf_parquet=CacheHFParquetDatasetConfig(**hf_parquet_data) if hf_parquet_data else None,
        densefusion=CacheDenseFusionDatasetConfig(**densefusion_data) if densefusion_data else None,
    )

    return CacheConfig(
        dataset=dataset_cfg,
        processing=CacheProcessingConfig(**data["processing"]),
        caption_text_lookup=CacheTextLookupConfig(**caption_lookup_data),
        alt_text_lookup=CacheTextLookupConfig(**alt_lookup_data),
        output=CacheOutputConfig(**data["output"]),
        vae=VAEConfig(**data["vae"]),
        dataloader=CacheDataloaderConfig(**dataloader_data) if dataloader_data else CacheDataloaderConfig(),
        aspect_ratio_bucket=int(data.get("aspect_ratio_bucket", 256)),
    )
