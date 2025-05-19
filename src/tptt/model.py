import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from .injection import inject_linear_attention
from .tuner import AdjustMaGWeightCallback
from .utils import instruction_format


class TpttConfig:
    """Configuration for TPTT model."""

    def __init__(
        self,
        model_name: str,
        target_modules_names: str = "self_attn",
        operator_mode: str = "delta_rule",
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
    ):
        """Initialize the TPTT configuration."""
        self.model_name = model_name
        self.target_modules_names = target_modules_names
        self.operator_mode = operator_mode
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size


class TpttModel:
    """Transforming Pretrained Transformer into Titans."""

    def __init__(
        self,
        model_name: str,
        config: TpttConfig,
    ):
        """Initialize the TpttModel with the given parameters."""
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = config
        self.get_target_modules()
        self.inject_liza_attention()

    def get_target_modules(self):
        """Load the model and tokenizer."""
        self.target_modules = [
            name
            for name, module in self.model.named_modules()
            if name.endswith(self.config.target_modules_names)
        ]

    def inject_liza_attention(self):
        """Inject LiZA attention into the model."""
        self.model = inject_linear_attention(
            self.model,
            self.model.config,
            target_modules=self.target_modules,
            operator_mode=self.config.operator_mode,
            mag_weight=self.config.mag_weight,
            max_chunk_size=self.config.max_chunk_size,
        )

    def generate(self, prompt: str):
        """Generate text using the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=50, do_sample=False  # deterministic
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def inject_lora_parameters(
        self,
        peft_config: LoraConfig = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    ):
        """Inject LoRA parameters into the model."""
        self.model = get_peft_model(self.model, peft_config)

    def tokenize(self, sample):
        tokens = self.tokenizer(
            sample["text"], truncation=True, max_length=256, padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def train(
        self,
        dataset=load_dataset("yahma/alpaca-cleaned")["train"].select(range(1000)),
        instruction_format=instruction_format,
        training_args: TrainingArguments = TrainingArguments(
            per_device_train_batch_size=2,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="tensorboard",
        ),
    ):
        """Train the model with the given dataset."""
        dataset = dataset.map(instruction_format)
        tokenized_dataset = dataset.map(self.tokenize, batched=True)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
        trainer.train()

    def save_model(self, path: str):
        """Save the model to the specified path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


if __name__ == "__main__":
    # Example usage
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tptt_model = TpttModel(model_name)

    prompt = "Once upon a time,"
    generated_text = tptt_model.generate(prompt)
    print(generated_text)

    # Save the model
    tptt_model.save_model("./liza_llama-instruct_model")
