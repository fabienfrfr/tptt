#!/usr/bin/env python3
import typer
import yaml
from pathlib import Path
import torch
import psutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

DTYPE_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

app = typer.Typer()
CONFIG_PATH = Path("titanesque_v2_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def estimate_memory_mistral(batch_size: int, model_size: int, quantization: bool):
    base_ft_mem_gb = 52.0  # base fine-tuning memory for Mistral 7B
    activation_per_batch_gb = 7.0
    base_mem_scaled = base_ft_mem_gb * (model_size / 7_000_000_000)
    total_mem = base_mem_scaled + batch_size * activation_per_batch_gb
    if quantization:
        total_mem *= 0.25
    return total_mem


def estimate_memory_usage(cfg):
    model_size = cfg.get("model_size", 7_000_000_000)
    quantization = cfg.get("quantized", False)
    batch_size = int(
        cfg.get("batch_size", 1) if cfg.get("batch_size", "auto") != "auto" else 1
    )
    seq_len = int(cfg.get("max_seq_len", 386))
    model_type = cfg.get("model_type", "causal")
    bidirectional = cfg.get("bidirectional", False)
    operator = cfg.get("operator", "")
    device = cfg.get("device", "cuda:0")

    if model_type == "mistral":
        total_mem_gb = estimate_memory_mistral(batch_size, model_size, quantization)
    else:
        dtype = torch.bfloat16 if not quantization else torch.float16
        param_mem_bytes = model_size * DTYPE_SIZE[dtype]
        activation_mem_bytes = batch_size * seq_len * 4 * DTYPE_SIZE[dtype] * 20
        total_mem_bytes = param_mem_bytes + activation_mem_bytes
        optimizer_overhead = param_mem_bytes * 1.2
        total_mem_bytes += optimizer_overhead
        total_mem_gb = total_mem_bytes / (1024**3)

    if bidirectional:
        total_mem_gb *= 2
    if "delta_product" in operator:
        total_mem_gb *= 2

    if torch.cuda.is_available() and "cuda" in device:
        try:
            gpu_index = int(device.split(":")[1])
        except Exception:
            gpu_index = 0
        props = torch.cuda.get_device_properties(gpu_index)
        available_mem_gb = props.total_memory / (1024**3)
        device_name = props.name
    else:
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)
        device_name = "CPU"

    print(f"🔍 Memory estimation for {model_type} (batch size {batch_size})")
    print(f"   Quantization    : {quantization}")
    print(f"   Device          : {device_name}")
    print(f"   Required memory : {total_mem_gb:.2f} GB")
    print(f"   Available memory: {available_mem_gb:.2f} GB")

    suggestions = []
    if total_mem_gb > available_mem_gb:
        suggestions.append("Reduce batch size (by half or more)")
        suggestions.append("Enable quantization (--quantization True)")
        suggestions.append("Use a smaller model")
        suggestions.append("Use different hardware (GPU with more memory)")

    if suggestions:
        print("\n⚠️ Not enough memory. Suggestions:")
        for s in suggestions:
            print(f"  - {s}")
    else:
        print("✅ Memory configuration OK.")

    return total_mem_gb, available_mem_gb, suggestions


def apply_delta_method(model, method, mag_weight, disable_linear_attn, liza_mode):
    """
    Apply delta method; replace stubs with your real implementations.
    """
    print(
        f"🔧 Applying {method} (mag_weight={mag_weight}) "
        f"{'with linear_attn disabled' if disable_linear_attn else ''} "
        f"{f'liza_mode={liza_mode}' if liza_mode else ''}"
    )
    # Replace with your delta methods:
    # if method == "delta_rule":
    #     return apply_delta_rule(model, mag_weight, disable_linear_attn, liza_mode)
    # elif method == "delta_product":
    #     return apply_delta_product(model, mag_weight, disable_linear_attn, liza_mode)
    # elif method == "delta_product_derived":
    #     return apply_delta_product_derived(model, mag_weight, disable_linear_attn, liza_mode)
    # else:
    #     raise ValueError(f"Unknown delta method: {method}")
    return model


def apply_lora(model, target_modules, r=8, alpha=16, dropout=0.05, bias="none"):
    """
    Apply LoRA using PEFT; replace stub accordingly.
    """
    print(f"🔧 Applying LoRA on modules: {target_modules}")
    # Example using PEFT:
    # lora_cfg = LoraConfig(
    #     r=r,
    #     lora_alpha=alpha,
    #     target_modules=target_modules,
    #     lora_dropout=dropout,
    #     bias=bias,
    #     task_type="CAUSAL_LM"
    # )
    # model = get_peft_model(model, lora_cfg)
    return model


@app.command()
def train(
    model_name: str = typer.Option(..., help="Model name from YAML"),
    method: str = typer.Option(
        ..., help="Delta method: delta_rule / delta_product / delta_product_derived"
    ),
    mag_weight: float = typer.Option(0.5, help="Magnitude weight"),
    disable_linear_attn: bool = typer.Option(False, help="Disable linear attention"),
    lora: bool = typer.Option(True, help="Enable LoRA"),
    liza_mode: str = typer.Option(
        "", help="Liza callback mode, e.g., constant / transition_X / alt_A_B"
    ),
    quantization: bool = typer.Option(False, help="Quantize model weights"),
    device: str = typer.Option("cuda:0", help="Device for training"),
):
    """
    Unified training script loading parameters from YAML and CLI, performs training and evaluation.
    """
    cfg = load_config()
    defaults = cfg["defaults"]

    model_entry = next((m for m in cfg["models"] if m["name"] == model_name), None)
    if not model_entry:
        typer.echo(f"❌ Model '{model_name}' not found in YAML configuration")
        raise typer.Exit(code=1)

    tokenizer_name = model_entry.get("tokenizer", model_name)
    target_modules = model_entry.get(
        "target_modules",
        cfg.get("lora_config", {}).get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    model_type = model_entry.get("type", "causal")

    # Estimate memory requirements
    memory_cfg = {
        "batch_size": (
            defaults.get("batch_size", 1)
            if defaults.get("batch_size", "auto") != "auto"
            else 1
        ),
        "model_size": 7_000_000_000,  # Optionally use actual model parameter count here
        "quantized": quantization or model_entry.get("quantized", False),
        "max_seq_len": defaults.get("max_length", 386),
        "model_type": model_type,
        "bidirectional": model_entry.get("bidirectional", False),
        "operator": method,
        "device": device,
    }
    estimate_memory_usage(memory_cfg)

    typer.echo(f"🚀 Training model {model_name} with {method}, mag_weight={mag_weight}")

    dataset = load_dataset(defaults["dataset"])
    train_len = int(len(dataset["train"]) * defaults["dataset_percentage"])
    train_ds = dataset["train"].shuffle(seed=defaults["seed"]).select(range(train_len))
    test_ds = dataset["train"].select(range(defaults["test_samples"]))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_func(batch):
        return tokenizer(
            batch["text"],
            padding=defaults["padding"],
            truncation=True,
            max_length=defaults["max_length"],
        )

    tokenized_train = train_ds.map(tokenize_func, batched=True)
    tokenized_test = test_ds.map(tokenize_func, batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(
            torch.bfloat16
            if defaults.get("linear_precision", "bfloat16") == "bfloat16"
            and not quantization
            else torch.float16
        ),
        device_map="auto",
    )

    if lora:
        model = apply_lora(
            model,
            target_modules,
            r=cfg.get("lora_config", {}).get("r", 8),
            alpha=cfg.get("lora_config", {}).get("lora_alpha", 16),
            dropout=cfg.get("lora_config", {}).get("lora_dropout", 0.05),
            bias="none",
        )

    model = apply_delta_method(
        model, method, mag_weight, disable_linear_attn, liza_mode
    )

    training_cfg = cfg.get("training_config", {}).get(
        defaults.get("training_config", "standard"), {}
    )
    args = TrainingArguments(
        output_dir=defaults["save_model_dir"],
        per_device_train_batch_size=int(memory_cfg["batch_size"]),
        num_train_epochs=training_cfg.get("epochs", 1),
        learning_rate=training_cfg.get("learning_rate", 5e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        gradient_accumulation_steps=training_cfg.get("grad_accumulation_steps", 1),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        logging_dir=defaults.get("save_logs_dir", "./logs"),
        logging_steps=training_cfg.get("logging_steps", 5),
        save_steps=training_cfg.get("save_steps", 1000),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        bf16=cfg.get("precision", {}).get("bf16", True),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "steps"),
        seed=defaults["seed"],
        report_to=training_cfg.get("report_to", "tensorboard"),
        fp16=cfg.get("precision", {}).get("fp16", False),
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()

    if defaults.get("run_test_eval", True):
        metrics = trainer.predict(tokenized_test).metrics
        typer.echo(f"📊 Test metrics: {metrics}")


if __name__ == "__main__":
    app()
