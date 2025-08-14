"""Some code for prepare and push titans model to hub destination"""

import os
import re
from typing import Optional
from dataclasses import dataclass, field

import yaml
import psutil
import torch
import typer

DTYPE_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), "r"
) as f:
    config = yaml.safe_load(f)


MODEL_CONFIGS = config.get("model_configs", {})
TRAINING_CONFIGS = config.get("training_configs", {})


def extract_model_size(base_model: str) -> int:
    """
    Extract the model size in number of parameters from a base_model string.
    The function looks for the last number before 'B' and interprets it as billions of parameters.
    """
    if "B" not in base_model:
        raise ValueError(f"Could not find 'B' in: {base_model}")

    before_b = base_model.rsplit("B", 1)[0]

    parts = re.split(r"[^0-9._]", before_b)
    numbers = [p for p in parts if p.strip()]
    last_num_str = numbers[-1].replace("_", ".").replace("-", ".")

    num_val = float(last_num_str)
    return int(num_val * 1_000_000_000)


@dataclass
class TrainConfig:
    # --- Model parameters ---
    base_model: Optional[str] = "meta-llama/Llama-3.2-1B"
    model_type: str = "mistral"
    batch_size: int = 2
    quantization: Optional[bool] = False
    model_size: Optional[int] = None
    base_tokenizer: Optional[str] = None
    target_modules: Optional[list[str]] = None

    # --- Hardware ---
    device: Optional[str] = None

    # --- General ---
    operator: str = "delta_rule"
    bidirectional: bool = False
    max_chunk_size: int = 64
    linear_precision: str = "bfloat16"
    base_attention_scaling: bool = False

    # --- MAG parameters ---
    n_transition_steps: int = 10
    initial_mag: float = 0.0
    mag_ratio: float = 0.5
    base_scale: bool = False
    mag_mode: str = "gradual"
    cross_gate: bool = False

    # --- LoRA parameters ---
    lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # --- Training hyperparameters ---
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    label_smooth: float = 0.0
    accumulation_grad_step: int = 1
    epoch: int = 1

    # --- Dataset parameters ---
    n_samples: int = 100
    test_eval_percent: int = 2
    dataset: str = "yahma/alpaca-cleaned"
    max_seq_len: int = 256

    # --- Hugging Face Hub parameters ---
    hf_token: Optional[str] = None
    username: str = "ffurfaro"
    repo_name: str = "Titans-v2-model"

    def __post_init__(self):
        self.load_presets()

    def load_presets(self):
        """Load defaults from config.yaml and recompute dependent fields."""
        config = MODEL_CONFIGS.get(self.model_type.lower())
        train_config = TRAINING_CONFIGS.get(self.model_type.lower(), {})

        if config is None:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        # Model params
        self.base_model = self.base_model or config.get("base_model_name")
        self.base_tokenizer = self.base_tokenizer or config.get("base_tokenizer_name")
        self.target_modules = self.target_modules or config.get("target_modules", [])
        self.quantization = (
            self.quantization
            if self.quantization is not None
            else config.get("quantization", False)
        )
        self.model_size = self.model_size or extract_model_size(self.base_model)

        # Hardware
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Training params
        self.batch_size = self.batch_size or train_config.get("batch_size", 2)
        self.learning_rate = self.learning_rate or train_config.get(
            "learning_rate", 5e-4
        )
        self.epoch = self.epoch or train_config.get("epoch", 1)
        self.lora = (
            self.lora if self.lora is not None else train_config.get("lora", True)
        )
        self.lora_rank = self.lora_rank or train_config.get("lora_rank", 8)
        self.lora_alpha = self.lora_alpha or train_config.get("lora_alpha", 16)
        self.lora_dropout = self.lora_dropout or train_config.get("lora_dropout", 0.05)

        # Dataset defaults
        self.max_seq_len = self.max_seq_len or train_config.get("max_seq_len", 256)
        self.dataset = self.dataset or train_config.get(
            "dataset", "yahma/alpaca-cleaned"
        )

        # HF Hub defaults
        self.username = self.username or train_config.get("username", "default-user")
        self.repo_name = self.repo_name or train_config.get(
            "repo_name", f"{self.model_type}-model"
        )

    def update_from_dict(self, params: dict):
        """Update config values from a dict and reapply linked logic."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown config key: {key}")
        # After updating, reload presets so all dependent fields get recomputed
        self.load_presets()


def estimate_memory_mistral(batch_size: int, model_size: int, quantization: bool):
    """
    Estimates GPU memory (GB) needed for Mistral fine-tuning (16-bit),
    Slidding Windows Attention (SWA) change memory calculation
    """
    # Reference base fine-tuning memory for Mistral 7B
    base_ft_mem_gb = 52.0  # average between 50 and 55 GB

    # Activation + optimizer overhead per batch (estimated higher than inference)
    activation_per_batch_gb = 7.0  # rough estimate for fine-tuning

    # Scale base memory by model size proportionally (default 7B)
    base_mem_scaled = base_ft_mem_gb * (model_size / 7_000_000_000)

    # Total memory estimate
    total_mem = base_mem_scaled + batch_size * activation_per_batch_gb

    # Quantization reduces memory roughly by factor 4
    if quantization:
        total_mem *= 0.25

    return total_mem


def estimate_memory_usage(cfg: TrainConfig):
    """
    Estimate GPU memory (GB) needed for training given config,
    compares to device availability and suggests adjustments if needed.
    """

    model_size = cfg.model_size
    quantization = cfg.quantization
    batch_size = cfg.batch_size
    seq_len = cfg.max_seq_len

    if cfg.model_type == "mistral":
        total_mem_gb = estimate_memory_mistral(batch_size, model_size, quantization)
    else:
        # Generic estimation for other models based on params, dtype, batch, seq_len
        dtype = torch.bfloat16 if not quantization else torch.float16
        param_mem_bytes = model_size * DTYPE_SIZE[dtype]

        activation_mem_bytes = (
            batch_size * seq_len * 4 * DTYPE_SIZE[dtype] * 20
        )  # rough estimate

        total_mem_bytes = param_mem_bytes + activation_mem_bytes

        # Extra 20% overhead for optimizer states, LoRA, etc.
        optimizer_overhead = param_mem_bytes * 1.2
        total_mem_bytes += optimizer_overhead

        total_mem_gb = total_mem_bytes / (1024**3)

    if cfg.bidirectional:
        # Bidirectional models typically double the memory usage
        total_mem_gb *= 2
    if "delta_product" in cfg.operator:
        # Delta product operator may increase memory usage
        total_mem_gb *= 2

    # Detect available memory on device
    if torch.cuda.is_available() and "cuda" in cfg.device:
        try:
            gpu_index = int(cfg.device.split(":")[1])
        except Exception:
            gpu_index = 0
        props = torch.cuda.get_device_properties(gpu_index)
        available_mem_gb = props.total_memory / (1024**3)
        device_name = props.name
    else:
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)
        device_name = "CPU"

    print(f"🔍 Memory estimation for {cfg.model_type} (batch size {batch_size})")
    print(f"   Quantization    : {quantization}")
    print(f"   Device          : {device_name}")
    print(f"   Required memory : {total_mem_gb:.2f} GB")
    print(f"   Available memory: {available_mem_gb:.2f} GB")

    suggestions = []
    if total_mem_gb > available_mem_gb:
        suggestions.append("Reduce batch size (by half or more)")
        suggestions.append("Enable quantization (--quantization True)")
        suggestions.append("Use a smaller model")
        suggestions.append("Use a different hardware (GPU with more memory)")

    if suggestions:
        print("\n⚠️ Not enough memory. Suggestions:")
        for s in suggestions:
            print(f"  - {s}")
    else:
        print("✅ Memory configuration OK.")

    return total_mem_gb, available_mem_gb, suggestions


app = typer.Typer(add_completion=False)


@app.command()
def check_memory(
    model_type: str = typer.Option(
        ..., help="Type of model: llama, mistral, olmo, etc."
    ),
    batch_size: int = typer.Option(2, help="Batch size for training"),
    quantization: bool = typer.Option(False, help="Whether to use quantization"),
    device: Optional[str] = typer.Option(
        None, help="Device to use, e.g. 'cuda:0' or 'cpu'"
    ),
    model_size: Optional[int] = typer.Option(
        None, help="Optional: model size in parameters (overrides auto detection)"
    ),
):
    """
    Check GPU memory needed for fine-tuning the given model and suggest adjustments.
    """
    # Import your TrainConfig and estimate_memory_usage from your module here
    # from your_module import TrainConfig, estimate_memory_usage

    cfg = TrainConfig(
        model_type=model_type,
        batch_size=batch_size,
        quantization=quantization,
        device=device,
        model_size=model_size,
    )
    estimate_memory_usage(cfg)


if __name__ == "__main__":
    # !python scripts/training_config.py check_memory ?
    app()
