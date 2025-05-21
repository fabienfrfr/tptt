"""This module implements the TPTT model with linear attention and LoRA support."""

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Pipeline,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from .injection import inject_linear_attention
from .liza.memory_gate import LiZAttention
from .tuner import AdjustMaGWeightCallback
from .utils import Cache, instruction_format


class TpttConfig(PretrainedConfig):
    """Configuration class for the TPTT model."""

    model_type = "tptt"

    def __init__(
        self,
        base_model_name="gpt2",
        base_tokenizer_name=None,
        target_modules_names=None,
        operator_mode="delta_rule",
        mag_weight=0.5,
        max_chunk_size=64,
        **kwargs,
    ):
        """
        Initialize TpttConfig with model and attention parameters.

        Args:
            base_model_name (str): Name of the base model.
            base_tokenizer_name (str): Name of the tokenizer.
            target_modules_names (list): List of module name suffixes to target.
            operator_mode (str): Operator mode for attention.
            mag_weight (float): Weight for MaG.
            max_chunk_size (int): Maximum chunk size for attention.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        if target_modules_names is None:
            target_modules_names = ["attn", "self_attn", "attention"]
        self.base_model_name = base_model_name
        self.base_tokenizer_name = (
            base_model_name if base_tokenizer_name is None else base_tokenizer_name
        )
        self.target_modules_names = target_modules_names
        self.operator_mode = operator_mode
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size


class TpttModel(PreTrainedModel):
    """TPTT model wrapper with linear attention and LoRA support."""

    config_class = TpttConfig

    def __init__(self, config: TpttConfig):
        """
        Initialize TpttModel.

        Args:
            config (TpttConfig): Model configuration.
        """
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_tokenizer_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.cache = Cache()
        self._inject_liza_attention()

    def _inject_liza_attention(self):
        """Inject LiZAttention into target modules."""
        target_modules = [
            name
            for name, _ in self.model.named_modules()
            if any(name.endswith(suffix) for suffix in self.config.target_modules_names)
        ]
        if not target_modules:
            raise ValueError(
                f"Target modules '{self.config.target_modules_names}' not found in the model."
            )
        self.model, self.cache = inject_linear_attention(
            self.model,
            self.model.config,
            liza_attention=LiZAttention,
            target_modules=target_modules,
            operator_mode=self.config.operator_mode,
            mag_weight=self.config.mag_weight,
            max_chunk_size=self.config.max_chunk_size,
        )

    def add_lora(self, lora_config: LoraConfig = None):
        """
        Add LoRA adapters to the model.

        Args:
            lora_config (LoraConfig, optional): LoRA configuration.
        """
        if lora_config is None:
            candidate_names = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # Llama, Mistral, OLMo
                "qkv_proj",
                "out_proj",  # OpenELM, some GPTs
                "c_attn",
                "c_proj",  # GPT-2
            ]
            target_modules = [
                name
                for name, _ in self.model.named_modules()
                if any(name.endswith(n) for n in candidate_names)
            ]
            target_modules = list(set(target_modules))
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)

    def save_pretrained(self, path: str, **kwargs):
        """Save model, tokenizer, and config to the given path."""
        self.model.save_pretrained(path, **kwargs)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model, tokenizer, and config from the given path."""
        config = TpttConfig.from_pretrained(path)
        obj = cls(config)
        obj.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        return obj


class TpttTrainer:
    """Trainer for TPTT models."""

    def __init__(
        self,
        model: TpttModel,
        tokenized_dataset=None,
        training_args=None,
        initial_weight=0.01,
        final_weight=0.5,
        transition_step=500,
    ):
        """
        Initialize TpttTrainer.

        Args:
            model (TpttModel): The TPTT model.
            tokenized_dataset (Dataset, optional): Pre-tokenized dataset.
            training_args (TrainingArguments, optional): Training arguments.
            initial_weight (float): Initial MaG weight.
            final_weight (float): Final MaG weight.
            transition_step (int): Transition step for MaG weight.
        """
        self.model = model
        self.tokenizer = model.tokenizer

        if tokenized_dataset is None:
            raw_dataset = load_dataset("yahma/alpaca-cleaned")["train"].map(
                instruction_format
            )
            self.tokenized_dataset = raw_dataset.map(
                self.tokenize, batched=True, remove_columns=raw_dataset.column_names
            )
        else:
            self.tokenized_dataset = tokenized_dataset

        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, padding="max_length", return_tensors="pt"
        )  # padding="longest"

        self.training_args = training_args or TrainingArguments(
            output_dir="./tptt_output",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="tensorboard",
        )

        self.liza_callback = AdjustMaGWeightCallback(
            self.model.model,
            initial_weight=initial_weight,
            final_weight=final_weight,
            transition_step=transition_step,
        )

    def tokenize(self, samples):
        """
        Tokenize samples in batch.

        Args:
            samples (dict): Batch of samples.

        Returns:
            dict: Tokenized samples with labels.
        """
        tokens = self.tokenizer(
            samples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",  # "longest",
            return_attention_mask=True,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def train(self, trainer=None):
        """
        Train the model.

        Args:
            trainer (Trainer, optional): Custom trainer instance.
        """
        trainer = (
            Trainer(
                model=self.model.model,
                args=self.training_args,
                train_dataset=self.tokenized_dataset,
                data_collator=self.data_collator,
                callbacks=[self.liza_callback],
                tokenizer=self.tokenizer,
            )
            if trainer is None
            else trainer
        )
        trainer.train()


class TpttPipeline(Pipeline):
    """Pipeline for TPTT model inference."""

    def __init__(self, model: TpttModel):
        """
        Initialize TpttPipeline.

        Args:
            model (TpttModel): The TPTT model.
        """
        super().__init__(model=model.model, tokenizer=model.tokenizer)
        self.model_wrapper = model

    def __call__(self, prompt, **kwargs):
        """
        Generate output from the model given a prompt.

        Args:
            prompt (str): Input prompt.
            **kwargs: Additional generation arguments.

        Returns:
            str: Generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 50),
                do_sample=kwargs.get("do_sample", False),
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
