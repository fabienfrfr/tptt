#!/usr/bin/env python3
"""PROTOTYPING SCRIPT
This script is used to train a model with the tptt framework.
But it for testing and prototyping purposes.
It is not intended for production use. (squeleton code)
"""
import os
import re
import shutil
import typer
import yaml
from pathlib import Path
import numpy as np

# force to avoid fragmentation error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import psutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from huggingface_hub import HfApi
from dotenv import load_dotenv

# install tptt with `pip install tptt`
import tptt


# Global
api = HfApi()
load_dotenv()

DTYPE_SIZE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

app = typer.Typer()

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "titanesque_config.yaml"
# CONFIG_PATH = Path("tptt/configs/titanesque_config.yaml")


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


def dummy_generate(model, tokenizer, prompt="Bonjour, I'm Fabien Furfaro,"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.linear_cache.reset()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device).eval()
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )
    print(tokenizer.decode(outputs[0]))


def remove_checkpoint_folders(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            shutil.rmtree(item_path)
            print(f"Deleted : {item_path}")


def extract_logs_from_trainer(configs, trainer):
    last_log = trainer.state.log_history[-2]
    train_log = trainer.state.log_history[-1]
    metrics = {
        "loss": last_log.get("loss", train_log.get("train_loss", "N/A")),
        "learning_rate": last_log.get("learning_rate", "N/A"),
        "epochs": train_log.get("epoch", "N/A"),
        "train_runtime": train_log.get("train_runtime", "N/A"),
        "train_samples_per_second": train_log.get("train_samples_per_second", "N/A"),
        "train_steps_per_second": train_log.get("train_steps_per_second", "N/A"),
        "total_flos": train_log.get("total_flos", "N/A"),
        "grad_norm": last_log.get("grad_norm", "N/A"),
    }

    return {
        "batch_size": configs["batch_size"],
        "dataset": configs["dataset"],
        "platform": configs["platform"],
        "hardware": configs["hardware_target"],
        **metrics,
    }


def estimate_memory_mistral(batch_size: int, model_size: int, quantization: bool):
    base_ft_mem_gb = 52.0
    activation_per_batch_gb = 7.0
    base_mem_scaled = base_ft_mem_gb * (model_size / 7_000_000_000)
    total_mem = base_mem_scaled + batch_size * activation_per_batch_gb
    if quantization:
        total_mem *= 0.25
    return total_mem


def estimate_memory_usage(cfg, batch_size=None):
    model_size = extract_model_size(cfg.get("model_name"))
    quantization = cfg.get("model_quantized", False)
    batch_size = batch_size if batch_size is not None else 1
    seq_len = int(cfg.get("max_length", 386))
    model_task = cfg.get("model_task", "causal_lm")
    bidirectional = cfg.get("bidirectional", False)
    operator = cfg.get("operator", "")

    if model_task == "mistral":  # change to "mistral" in model_name
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

    if torch.cuda.is_available():
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
    batch_size = min(max(1, int(0.75 * available_mem_gb // total_mem_gb)), max_batch)
    print(f"🔥 Auto batch sizing is {batch_size}, based on memory usage ratio")
    return batch_size


@app.command()
def train(
    config: Path = typer.Option(CONFIG_PATH, help="Path to the YAML conf"),
    model_name: str = typer.Option(None, help="Model name, overrides config"),
    model_task: str = typer.Option(None, help="Model type, overrides config"),
    dataset: str = typer.Option(None, help="Dataset name, overrides config"),
    lora: bool = typer.Option(None, help="Enable LoRA training"),
    operator_mode: str = typer.Option(None, help="Delta method"),
    liza_callback_mode: str = typer.Option(None, help="Liza callback mode"),
    liza_final_weight: float = typer.Option(None, help="Memory gate weight"),
    model_quantized: bool = typer.Option(None, help="Enable quantization model"),
    use_linear_checkpoint: bool = typer.Option(None, help="RAM linear layers"),
    token: str = typer.Option(None, help="HuggingFace token access"),
    max_length: int = typer.Option(None, help="Maximum sequence len tokenization"),
    padding: str = typer.Option(None, help="Padding strategy for tokenization"),
    batch_size: int = typer.Option(None, help="'auto' for automatic sizing"),
    lora_r: int = typer.Option(None, help="LoRA r rank"),
    lora_alpha: int = typer.Option(None, help="LoRA alpha value"),
    lora_task_type: str = typer.Option(None, help="LoRA task type"),
    lora_dropout: float = typer.Option(None, help="LoRA dropout rate"),
    model_tokenizer: str = typer.Option(None, help="Tokenizer model name"),
    lora_target_modules: str = typer.Option(None, help="Comma list of LoRA target"),
    max_chunk_size: int = typer.Option(None, help="Maximum chunk size for model"),
    cross_gate_mode: bool = typer.Option(None, help="Whether to use cross gate mode"),
    linear_precision: str = typer.Option(None, help="Linear layer precision"),
    home_model_dir: str = typer.Option(None, help="Home directory to save model"),
    folder_model_dir: str = typer.Option(None, help="Directory to save trained model"),
    training_learning_rate: float = typer.Option(None, help="Training learning rate"),
    training_epochs: int = typer.Option(None, help="Number of training epochs"),
    training_weight_decay: float = typer.Option(None, help="decay for optimizer"),
    training_gradient_accumulation_steps: int = typer.Option(None),
    training_max_grad_norm: float = typer.Option(None, help="Maximum gradient norm"),
    training_label_smoothing_factor: float = typer.Option(None, help="Label smoothing"),
    training_logging_steps: int = typer.Option(None, help="Logging frequency"),
    training_save_steps: int = typer.Option(None, help="Checkpoint save frequency"),
    training_save_total_limit: int = typer.Option(None, help="number checkpoints"),
    training_bf16: bool = typer.Option(None, help="bf16 mixed precision"),
    training_evaluation_strategy: str = typer.Option(None, help="Evaluation strategy"),
    training_report_to: str = typer.Option(None, help="Reporting integration"),
    seed: int = typer.Option(None, help="Random seed"),
    dataset_percentage: float = typer.Option(None, help="Percentage of dataset"),
    eval_samples: int = typer.Option(None, help="Number of samples for validation"),
    test_samples: int = typer.Option(None, help="Number of samples for testing"),
    model_bidirectional: bool = typer.Option(None, help="If model bidirectional"),
    model_in_subfolder: bool = typer.Option(None, help="Save model in subfolder"),
    remove_checkpoint: bool = typer.Option(None, help="Flush checkpoints"),
    generating_model_card: bool = typer.Option(None, help="Flush checkpoints"),
    save_to_hfhub: bool = typer.Option(None, help="Save model to 🤗 hub"),
    repo_user: str = typer.Option(None, help="You're name in HuggingFace 🤗"),
    repo_name: str = typer.Option(None, help="Folder in HuggingFace 🤗"),
):

    print(
        f"🛠️ [WARNING] Prototyping scripts, to be used with caution or to reproduce results"
    )
    # prepare config
    configs = update_configs(**locals())
    # run training
    model, tokenizer, trainer = get_model_trainer(configs, token=token)
    print(f"🚀 Start training with this configs : {configs}")
    trainer.train()

    print(f"🚀 Start post-training !")

    # post training
    if configs["save_to_hfhub"]:
        save_to_huggingface(configs, tokenizer, model, trainer, token=token)

    # generate result training with dummy example
    dummy_generate(model, tokenizer)


def save_to_huggingface(configs, tokenizer, model, trainer, token=None):
    if configs["remove_checkpoint"]:
        remove_checkpoint_folders(configs["folder_model_dir"])

    tokenizer.save_pretrained(configs["folder_model_dir"])
    model.save_pretrained(
        configs["folder_model_dir"], subfolder=configs["model_in_subfolder"]
    )

    if configs["generating_model_card"]:
        train_vars = extract_logs_from_trainer(configs, trainer)

        # is flush if no subfolder
        tptt.generate_model_card(
            output_path=configs["home_model_dir"],
            config=model.config,
            template="home_card_template",
            extra_variables=train_vars,
        )

        tptt.generate_model_card(
            output_path=configs["folder_model_dir"],
            config=model.config,
            template="model_card_template",
            extra_variables=train_vars,
        )

    if configs["save_to_hfhub"]:
        hf_token = os.getenv("HF_TOKEN") if token is None else token
        repo_id = configs["repo_user"] + "/" + configs["repo_name"]
        api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        api.upload_folder(
            folder_path=configs["home_model_dir"],
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message="Upload model + init tptt code",
        )


def update_configs(config_path: Path = CONFIG_PATH, **kwargs):

    # Load config yaml then update with CLI arguments if present
    cfg = load_config(config_path)
    for k, v in kwargs.items():
        if v is not None and k not in ("config", "cfg"):
            cfg[k] = v
            if k != "token":
                print(f"🔥 Set key '{k}' to '{v}' of type {type(v).__name__} in config")

    # adapt save model path
    if cfg["home_model_dir"] == "":
        standards_repo_name = "Titanesque-" + cfg["model_name"].split("/")[-1]
        if cfg["repo_name"] == "":
            cfg["repo_name"] = standards_repo_name
        cfg["home_model_dir"] = "./" + standards_repo_name
        if cfg["model_in_subfolder"] and cfg["folder_model_dir"] == "":
            subfolder_name = (
                ("lora_" if cfg["lora"] else "")
                + ("cross_" if cfg["cross_gate_mode"] else "")
                + cfg["operator_mode"]
                + "_m"
                + str(cfg["liza_final_weight"])
                + "_"
                + cfg["liza_callback_mode"]
            )
            # true dirpath
            cfg["folder_model_dir"] = cfg["home_model_dir"] + "/" + subfolder_name
        else:
            cfg["folder_model_dir"] = cfg["home_model_dir"]

    print(f"🔥 Save model checkpoints and trainings in {cfg['folder_model_dir']}")

    cfg["batch_size"] = (
        cfg["batch_size"]
        if cfg["batch_size"] != "auto"
        else estimate_optimal_batch_size(cfg)
    )

    if isinstance(cfg["model_torch_dtype"], str):
        cfg["model_torch_dtype"] = getattr(torch, cfg["model_torch_dtype"])

    cfg["training_learning_rate"] = float(cfg["training_learning_rate"])

    print(
        f"🚀 Config for training model {cfg['model_name']} with {cfg['operator_mode']}, batch_size={cfg['batch_size']}"
    )

    cfg.pop("token", None)
    return cfg


def get_model_trainer(cfg: dict, token: str = None):
    # set in .env
    token = os.getenv("HF_TOKEN") if token is None else token

    # put in update config ?
    tokenizer_name = (
        cfg["model_tokenizer"] if cfg["model_tokenizer"] else cfg["model_name"]
    )
    target_modules = (
        cfg["lora_target_modules"]
        if cfg["lora_target_modules"]
        else ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Start trainer preparation
    dataset = load_dataset(cfg["dataset"])["train"]

    train_len = int(len(dataset) * cfg["dataset_percentage"])
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padding_side = tokenizer.padding_side

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

    lengths = [len(x["input_ids"]) for x in tokenized_datasets["train"]]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)

    print(f"🔥 TOKENIZED DATASET : mean={mean_length:.2f}, std={std_length:.2f} length")

    if cfg["lora"]:
        lora_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type=cfg["lora_task_type"],
            target_modules=target_modules,
        )
        lora_config_dict = lora_config.to_dict()
    else:
        lora_config_dict = None

    if cfg["model_quantized"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model_config = tptt.TpttConfig(
        base_model_name=cfg["model_name"],
        operator_mode=cfg["operator_mode"],
        max_self_attn_length=None,  # or use a config value if available
        model_task=cfg.get("model_task", "causal_lm"),
        max_chunk_size=cfg["max_chunk_size"],
        lora_config=lora_config_dict,
        mag_weight=cfg.get("liza_final_weight", 0.5),
        base_scale_attn=None,
        cross_gate=cfg["cross_gate_mode"],
        linear_precision=cfg["linear_precision"],
        use_linear_checkpoint=cfg["use_linear_checkpoint"],
        padding_side=padding_side,
        bidirectional=cfg.get("model_bidirectional", False),
        trust_remote_code=True,  # in kwargs for Autoconfig
    )

    model = tptt.TpttModel(
        model_config,
        trust_remote_code=True,
        torch_dtype=cfg.get("model_torch_dtype", torch.bfloat16),
        quantization_config=bnb_config,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=cfg["folder_model_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["training_epochs"],
        learning_rate=cfg["training_learning_rate"],
        weight_decay=cfg["training_weight_decay"],
        label_smoothing_factor=cfg["training_label_smoothing_factor"],
        gradient_accumulation_steps=cfg["training_gradient_accumulation_steps"],
        max_grad_norm=cfg["training_max_grad_norm"],
        # logging_dir="./logs",
        logging_steps=cfg["training_logging_steps"],
        # save_steps=cfg["training_save_steps"],
        ddp_find_unused_parameters=False,
        save_total_limit=cfg["training_save_total_limit"],
        bf16=cfg["training_bf16"],
        eval_strategy=cfg["training_evaluation_strategy"],
        seed=cfg["seed"],
        report_to=cfg["training_report_to"],
        # fp16=not cfg["training_bf16"],
        # push_to_hub=False,
        # disable_tqdm=False,
    )

    liza_callback = tptt.LiZACallback(
        model=model,
        mode=cfg.get("liza_callback_mode", "constant"),
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

    return model, tokenizer, trainer


if __name__ == "__main__":
    app()
