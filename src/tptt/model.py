import torch

from .config import TpttConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from .injection import inject_linear_attention
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


class TpttModel:
    """Transforming Pretrained Transformer into Titans."""

    def __init__(
        self,
        model_name: str,
        config: TpttConfig = TpttConfig(),
    ):
        """Initialize the TpttModel with the given parameters."""
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = config
        self.get_target_modules()
        self.inject_linear_attention()

    def get_target_modules(self):
        """Load the model and tokenizer."""
        self.target_modules = [
            name
            for name, module in self.model.named_modules()
            if name.endswith(self.config.target_modules_names)
        ]

    def inject_liza_attention(self):
        """Inject LiZA attention into the model."""
        self.model_ = inject_linear_attention(
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
            output = self.model_.generate(
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
        self.model_ = get_peft_model(self.model, peft_config)

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
        callbacks: list = [
            AdjustMaGWeightCallback(
                self.model_,
                initial_weight=0.01,
                final_weight=0.5,
                transition_step=500,
            )
        ],
    ):
        """Train the model with the given dataset."""
        dataset = dataset.map(instruction_format)
        tokenized_dataset = dataset.map(tokenize)
        Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
        trainer.train()

    def save_model(self, path: str):
        """Save the model to the specified path."""
        self.model_.save_pretrained(path)
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
