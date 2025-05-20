import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Pipeline,
                          PretrainedConfig, PreTrainedModel, Trainer,
                          TrainingArguments)

from .injection import inject_linear_attention
from .tuner import AdjustMaGWeightCallback
from .utils import Cache, instruction_format


# 1. CONFIGURATION
class TpttConfig(PretrainedConfig):
    model_type = "tptt"

    def __init__(
        self,
        base_model_name="gpt2",
        target_modules_names="self_attn",
        operator_mode="delta_rule",
        mag_weight=0.5,
        max_chunk_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.target_modules_names = target_modules_names
        self.operator_mode = operator_mode
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size


# 2. MODELE
class TpttModel(PreTrainedModel):
    config_class = TpttConfig

    def __init__(self, config: TpttConfig):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.cache = Cache()
        self._inject_liza_attention()

    def _inject_liza_attention(self):
        target_modules = [
            name
            for name, module in self.model.named_modules()
            if name.endswith(self.config.target_modules_names)
        ]
        self.model, self.cache = inject_linear_attention(
            self.model,
            self.model.config,
            target_modules=target_modules,
            operator_mode=self.config.operator_mode,
            mag_weight=self.config.mag_weight,
            max_chunk_size=self.config.max_chunk_size,
        )

    def add_lora(self, lora_config: LoraConfig = None):
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_pretrained(self, path: str, **kwargs):
        self.model.save_pretrained(path, **kwargs)
        self.tokenizer.save_pretrained(path)
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        config = TpttConfig.from_pretrained(path)
        obj = cls(config)
        obj.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        return obj


# 3. TRAINER
class TpttTrainer:
    def __init__(
        self,
        model: TpttModel,
        dataset,
        instruction_format_fn=instruction_format,
        training_args=None,
        initial_weight=0.01,
        final_weight=0.5,
        transition_step=500,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        self.dataset = dataset.map(instruction_format_fn)
        self.tokenized_dataset = self.dataset.map(self.tokenize)
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

    def tokenize(self, sample):
        tokens = self.tokenizer(
            sample["text"], 
            truncation=True,
            max_length=256, 
            padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def train(self):
        trainer = Trainer(
            model=self.model.model,
            args=self.training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            callbacks=[self.liza_callback],
        )
        trainer.train()


# 4. PIPELINE
class TpttPipeline(Pipeline):
    def __init__(self, model: TpttModel):
        super().__init__(model=model.model, tokenizer=model.tokenizer)
        self.model_wrapper = model

    def __call__(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 50),
                do_sample=kwargs.get("do_sample", False),
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    config = TpttConfig(
        base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )
    tptt_model = TpttModel(config)

    # Optionnal : LoRA
    tptt_model.add_lora()

    # Train
    dataset = load_dataset("yahma/alpaca-cleaned")["train"].select(range(1000))
    trainer = TpttTrainer(tptt_model, dataset)
    trainer.train()

    # Save
    tptt_model.save_pretrained("./liza_llama-instruct_model")

    # Inference pipeline
    pipeline = TpttPipeline(tptt_model)
    print(pipeline("Once upon a time,"))

    # Loading the model
    # tptt_model2 = TpttModel.from_pretrained("./liza_llama-instruct_model")
