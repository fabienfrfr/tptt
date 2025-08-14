"""Some code for train and push titans model to hub destination"""

import os
import torch
import typer
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig
import tptt
from huggingface_hub import HfApi

app = typer.Typer(add_completion=False)


# ========================
# 1️⃣ Training configuration data class
# ========================
@dataclass
class TrainConfig:
    # General
    quantization: bool = False
    model_type: str = "moe"
    operator: str = "delta_rule"
    bidirectional: bool = False

    # MAG parameters
    transition: int = 10
    initial_mag: float = 0.0
    mag_ratio: float = 0.5
    base_scale: bool = False
    mag_mode: str = "gradual"

    # LoRA parameters
    lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training hyperparameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    label_smooth: float = 0.0
    accumulation_grad_step: int = 1
    epoch: int = 1
    batch_size: int = 2

    # Dataset parameters
    n: int = 100
    test_eval_percent: int = 2
    dataset: str = "yahma/alpaca-cleaned"

    # Hugging Face Hub parameters
    hf_token: Optional[str] = None
    username: str = "ffurfaro"
    repo_name: str = "Titans-v2-model"

    # Hardware
    device: str = "cuda:0"


# ========================
# 2️⃣ Model resolution utility function
# ========================
def resolve_model_type(cfg: TrainConfig):
    """
    Map the chosen model_type to its base model name, tokenizer name and LoRA target modules.
    """
    mapping = {
        "moe": (
            "allenai/OLMoE-1B-7B-0924",
            "allenai/OLMoE-1B-7B-0924",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        "mistral": (
            "mistralai/Mistral-7B-v0.3",
            "mistralai/Mistral-7B-v0.3",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        "llama": (
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        "olmo": (
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-1B-hf",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    }
    return mapping[cfg.model_type]


# ========================
# 3️⃣ Complete training pipeline
# ========================
def train_model(cfg: TrainConfig):
    """
    Full training pipeline:
    - Load model and tokenizer
    - Prepare dataset
    - Train using HuggingFace Trainer
    - Save and push to Hugging Face Hub
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Resolve model, tokenizer and target modules based on model_type
    base_model_name, tokenizer_name, target_modules = resolve_model_type(cfg)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=cfg.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    # LoRA configuration
    if cfg.lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ).to_dict()
    else:
        lora_config = None

    # Quantization configuration
    bnb_config = None
    if cfg.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Model creation
    model = tptt.TpttModel(
        tptt.TpttConfig(
            base_model_name=base_model_name,
            operator_mode=cfg.operator,
            lora_config=lora_config,
            mag_weight=cfg.mag_ratio,
            base_scale_attn=cfg.base_scale,
            cross_gate=False,
            linear_precision=torch.bfloat16,
            bidirectional=cfg.bidirectional,
        ),
        trust_remote_code=True,
        token=cfg.hf_token,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    # Dataset preparation
    M = int(cfg.n * 100 / (100 - cfg.test_eval_percent))
    raw_dataset = load_dataset(cfg.dataset)["train"].select(range(M))
    split_dataset = raw_dataset.train_test_split(
        test_size=cfg.test_eval_percent / 100.0, seed=42
    )
    test_valid = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

    dataset = {
        "train": split_dataset["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"],
    }

    def preprocess_fn(samples):
        """
        Preprocess dataset by building the prompt and tokenizing it.
        """
        prompts = [
            f"{instr}\n{inp}" if inp else instr
            for instr, inp in zip(samples["instruction"], samples["input"])
        ]
        prompts = [f"{p}\n{out}" for p, out in zip(prompts, samples["output"])]
        tokens = tokenizer(
            prompts,
            truncation=True,
            max_length=256,
            padding="longest",
            return_attention_mask=True,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = {
        split: dataset[split].map(
            preprocess_fn, batched=True, remove_columns=dataset[split].column_names
        )
        for split in dataset
    }

    # Data collator for causal LM (MLM=False)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TrainingArguments setup
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epoch,
        learning_rate=cfg.learning_rate,
        bf16=True,
        logging_steps=5,
        label_smoothing_factor=cfg.label_smooth,
        weight_decay=cfg.weight_decay,
        gradient_accumulation_steps=cfg.accumulation_grad_step,
        evaluation_strategy="steps",
        report_to=None,
    )

    # LiZA Callback for MAG scheduling
    liza_callback = tptt.LiZACallback(
        model,
        initial_weight=cfg.initial_mag,
        final_weight=cfg.mag_ratio,
        transition_step=cfg.transition,
        mode=cfg.mag_mode,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[liza_callback],
    )

    trainer.train()

    # Save locally
    model.save_pretrained("./output", safe_serialization=False)
    tokenizer.save_pretrained("./output")

    # Push to Hugging Face Hub
    api = HfApi()
    repo_id = f"{cfg.username}/{cfg.repo_name}"
    api.create_repo(
        repo_id=repo_id, token=cfg.hf_token, repo_type="model", exist_ok=True
    )
    api.upload_folder(
        folder_path="./output",
        repo_id=repo_id,
        repo_type="model",
        token=cfg.hf_token,
        commit_message="Upload trained model",
    )

    typer.echo(f"✅ Training completed and pushed to HF Hub: {repo_id}")


# ========================
# 4️⃣ Typer CLI entrypoint
# ========================
@app.command()
def cli_train(
    model_type: str = "moe",
    operator: str = "delta_rule",
    dataset: str = "yahma/alpaca-cleaned",
    epoch: int = 1,
    learning_rate: float = 5e-4,
    hf_token: str = typer.Option(..., help="Hugging Face authentication token"),
):
    """
    Train the model from CLI.
    """
    cfg = TrainConfig(
        model_type=model_type,
        operator=operator,
        dataset=dataset,
        epoch=epoch,
        learning_rate=learning_rate,
        hf_token=hf_token,
    )
    train_model(cfg)


if __name__ == "__main__":
    app()
