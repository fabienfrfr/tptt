#!/usr/bin/env python3
import re
import typer
import yaml
import json
from pathlib import Path
import torch
import psutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import tptt

DTYPE_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

app = typer.Typer()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "titanesque_config.yaml"


def load_config(config):
    with open(config, "r") as f:
        return yaml.safe_load(f)


def extract_model_size(base_model: str) -> int:
    if "B" not in base_model:
        raise ValueError(f"Could not find 'B' in: {base_model}")
    before_b = base_model.rsplit("B", 1)[0]
    parts = re.split(r"[^0-9._]", before_b)
    numbers = [p for p in parts if p.strip()]
    last_num_str = numbers[-1].replace("_", ".").replace("-", ".")
    num_val = float(last_num_str)
    return int(num_val * 1_000_000_000)


def estimate_memory_mistral(batch_size: int, model_size: int, quantization: bool):
    base_ft_mem_gb = 52.0
    activation_per_batch_gb = 7.0
    base_mem_scaled = base_ft_mem_gb * (model_size / 7_000_000_000)
    total_mem = base_mem_scaled + batch_size * activation_per_batch_gb
    if quantization:
        total_mem *= 0.25
    return total_mem


def estimate_memory_usage(cfg, batch_size=None):
    model_size = cfg.get("model_size", 7_000_000_000)
    quantization = cfg.get("quantized", False)
    batch_size = (
        batch_size
        if batch_size is not None
        else int(
            cfg.get("batch_size", 1) if cfg.get("batch_size", "auto") != "auto" else 1
        )
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
    else:
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)
    return total_mem_gb, available_mem_gb


def estimate_optimal_batch_size(cfg, max_batch=64):
    batch_size = 1
    total_mem_gb, available_mem_gb = estimate_memory_usage(cfg, batch_size=batch_size)
    batch_size = min(max(1, int(available_mem_gb // total_mem_gb)), max_batch)
    return batch_size


@app.command()
def train(
    config: Path = typer.Option(
        CONFIG_PATH, help="Path to the configuration YAML file"
    ),
    model_name: str = typer.Option(None, help="Model name from YAML"),
    lora: bool = typer.Option(True, help="Enable LoRA"),
    method: str = typer.Option(
        ..., help="Delta method: delta_rule / delta_product / delta_product_derived"
    ),
    liza_mode: str = typer.Option(
        "", help="Liza callback mode, e.g., gradual / cyclic / switch"
    ),
    mag_weight: float = typer.Option(0.5, help="Magnitude weight"),
    quantization: bool = typer.Option(False, help="Quantize model weights"),
    extra_config: str = typer.Option("{}", help="Extra config as JSON string"),
):
    cfg = load_config(config)
    extra_config = json.loads(extra_config)
    cfg.update(extra_config)

    if model_name is not None:
        model_name = cfg["model_name"]

    tokenizer_name = cfg["model_tokenizer"] if cfg["model_tokenizer"] else model_name
    target_modules = (
        cfg["lora_target_modules"]
        if cfg["lora_target_modules"]
        else ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model_type = cfg.get("model_type", "causal")

    batch_size = (
        cfg["batch_size"]
        if cfg["batch_size"] != "auto"
        else estimate_optimal_batch_size(cfg)
    )

    memory_cfg = {
        "batch_size": batch_size,
        "quantized": quantization or cfg.get("model_quantized", False),
        "max_seq_len": cfg["max_length"],
        "model_type": model_type,
        "bidirectional": cfg.get("model_bidirectional", False),
        "operator": method,
    }
    estimate_memory_usage(memory_cfg)

    print(f"🚀 Training model {model_name} with {method}, batch_size={batch_size}")

    dataset = load_dataset(cfg["dataset"])

    train_len = int(len(dataset["train"]) * cfg["dataset_percentage"])
    eval_len = cfg.get("eval_samples", 2)
    test_len = cfg.get("test_samples", 2)

    test_eval_len = eval_len + test_len
    total_len = train_len + eval_len + test_len

    dataset = dataset.select(range(total_len))
    split_dataset = dataset.train_test_split(
        test_size=test_eval_len / total_len, seed=cfg["seed"], shuffle=True
    )
    test_validation_split = split_dataset["test"].train_test_split(
        test_size=test_len / test_eval_len, seed=cfg["seed"], shuffle=True
    )
    dataset = {
        "train": split_dataset["train"],
        "validation": test_validation_split["train"],
        "test": test_validation_split["test"],
    }

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_func(samples):
        prompts = [
            f"{instr}\n{inp}" if inp else instr
            for instr, inp in zip(samples["instruction"], samples["input"])
        ]
        prompts = [f"{p}\n{out}" for p, out in zip(prompts, samples["output"])]
        tokens = tokenizer(
            prompts,
            truncation=True,
            max_length=cfg["max_length"],
            padding=cfg["padding"],
            return_attention_mask=True,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_datasets = {}
    for split in ["train", "validation", "test"]:
        tokenized_datasets[split] = dataset[split].map(
            tokenize_func, batched=True, remove_columns=dataset[split].column_names
        )

    if lora:
        lora_config = tptt.LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        lora_config_dict = lora_config.to_dict()
    else:
        lora_config_dict = None

    model_config = tptt.TpttConfig(
        base_model_name=model_name,
        operator_mode=method,
        max_self_attn_length=None,  # or use a config value if available
        max_chunk_size=cfg["max_chunk_size"].get(method, 64),
        lora_config=lora_config_dict,
        mag_weight=mag_weight,
        base_scale_attn=None,
        cross_gate=cfg["cross_gate_mode"],
        linear_precision=cfg["linear_precision"],
        use_linear_checkpoint=False,
        padding_side=cfg["model_padding_side"],
        bidirectional=cfg.get("model_bidirectional", False),
    )

    model = tptt.TpttModel(
        model_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=None if not quantization else {},
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg["save_model_dir"],
        per_device_train_batch_size=memory_cfg["batch_size"],
        num_train_epochs=cfg["training_epochs"],
        learning_rate=cfg["training_learning_rate"],
        weight_decay=cfg["training_weight_decay"],
        gradient_accumulation_steps=cfg["training_gradient_accumulation_steps"],
        max_grad_norm=cfg["training_max_grad_norm"],
        logging_dir="./logs",
        logging_steps=cfg["training_logging_steps"],
        save_steps=cfg["training_save_steps"],
        save_total_limit=cfg["training_save_total_limit"],
        bf16=cfg["training_bf16"],
        eval_strategy=cfg["training_evaluation_strategy"],
        seed=cfg["seed"],
        report_to=cfg["training_report_to"],
        fp16=not cfg["training_bf16"],
        push_to_hub=False,
    )

    liza_callback = tptt.LiZACallback(
        model=model,
        mode=liza_mode if liza_mode else cfg.get("liza_callback_mode", ""),
        initial_weight=cfg.get("liza_initial_weight", 0.0),
        final_weight=cfg.get("liza_final_weight", 0.5),
        transition_step=cfg.get("liza_transition_steps", 0),
        weight_list=cfg.get("liza_weight_list", [0.0, 0.5, 1.0]),
        switch_period=cfg.get("liza_switch_period", 1),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[liza_callback],
    )

    trainer.train()

    if cfg.get("run_test_eval", True):
        metrics = trainer.predict(tokenized_datasets["test"]).metrics
        typer.echo(f"📊 Test metrics: {metrics}")


if __name__ == "__main__":
    app()
