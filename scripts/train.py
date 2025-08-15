#!/usr/bin/env python3
import typer
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

app = typer.Typer()
CONFIG_PATH = Path("titanesque_v2_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def apply_delta_method(model, method, mag_weight, disable_linear_attn, liza_mode):
    """
    Stub for applying delta modifications — to be replaced with real code.
    """
    print(
        f"🔧 Applying {method} (mag_weight={mag_weight}) "
        f"{'with linear_attn disabled' if disable_linear_attn else ''} "
        f"{f'liza_mode={liza_mode}' if liza_mode else ''}"
    )
    # TODO: import your actual delta method implementation from scripts/
    return model


def apply_lora(model, target_modules):
    """
    Stub LoRA application — add PEFT or your LoRA library call here.
    """
    print(f"🔧 Applying LoRA on modules: {target_modules}")
    # TODO: integrate PEFT config
    return model


@app.command()
def train(
    model_name: str = typer.Option(..., help="Model from YAML"),
    method: str = typer.Option(
        ..., help="delta_rule / delta_product / delta_product_derived"
    ),
    mag_weight: float = typer.Option(0.5, help="Magnitude weight"),
    disable_linear_attn: bool = typer.Option(False, help="Disable linear attention"),
    lora: bool = typer.Option(True, help="Use LoRA"),
    liza_mode: str = typer.Option(
        "", help="Liza callback mode: constant / transition_X / alt_A_B"
    ),
):
    """
    Unified training script that loads config, applies delta methods, trains and evaluates a single config.
    """
    cfg = load_config()
    defaults = cfg["defaults"]

    model_entry = next((m for m in cfg["models"] if m["name"] == model_name), None)
    if not model_entry:
        typer.echo(f"❌ Model '{model_name}' not found in YAML")
        raise typer.Exit(code=1)

    typer.echo(f"🚀 Training [{model_name}] with {method} / mag_weight={mag_weight}")

    # Load dataset
    ds = load_dataset(defaults["dataset"])
    train_ds = (
        ds["train"]
        .shuffle(seed=defaults["seed"])
        .select(range(int(len(ds["train"]) * defaults["dataset_percentage"])))
    )
    test_ds = ds["train"].select(range(defaults["test_samples"]))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tok(batch):
        return tokenizer(
            batch["text"],
            padding=defaults["padding"],
            truncation=True,
            max_length=defaults["max_length"],
        )

    tokenized_train = train_ds.map(tok, batched=True)
    tokenized_test = test_ds.map(tok, batched=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Apply LoRA
    if lora:
        tmodules = model_entry.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        model = apply_lora(model, tmodules)

    # Apply delta method
    model = apply_delta_method(
        model, method, mag_weight, disable_linear_attn, liza_mode
    )

    # Training args
    args = TrainingArguments(
        output_dir=defaults["save_model_dir"],
        per_device_train_batch_size=(
            1 if defaults["batch_size"] == "auto" else defaults["batch_size"]
        ),
        num_train_epochs=1,
        logging_dir=defaults["save_logs_dir"],
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=5,
        save_total_limit=1,
        seed=defaults["seed"],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    # Train
    trainer.train()

    # Eval
    if defaults["run_test_eval"]:
        metrics = trainer.predict(tokenized_test).metrics
        typer.echo(f"📊 Test metrics: {metrics}")


if __name__ == "__main__":
    app()
