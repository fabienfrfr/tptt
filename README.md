# LiZA: LineariZe Attention Injection

LiZA is a modular Python library designed to inject efficient linearized attention mechanisms-such as `Memory as Gate` (describes in [Titans](https://arxiv.org/html/2501.00663v1)) in pretrained transformers. 

It leverages the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library for high-performance implementations, enabling scalable and memory-efficient attention computations.


---

## Features

- **Flexible Attention Injection**: Seamlessly wrap and augment standard Transformer attention layers with linearized attention variants for latent memory.
- **Support for GLA and Delta Rule**: Includes implementations of Gated Linear Attention and Delta Rule Attention using flash-linear-attention.
- **Modular Design**: Easily extend or customize operators and integration strategies.
- **Typed Configuration**: Uses Pydantic for strict and clear configuration management.
- **Compatibility**: Designed to integrate with Hugging Face Transformers and similar PyTorch models.
- **Testing**: Comprehensive Pytest test suite with fixtures ensures reliability.

---

## Installation

```bash
git clone https://github.com/fabienfrfr/liza.git
cd liza
make install
```

*Note*: `flash-linear-attention` requires a CUDA-enabled GPU and compatible PyTorch version.

---

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from liza.injector import inject_linear_attention

# Charger votre modèle (ex: Llama, Mistral, etc.)
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Injecter l’attention linéaire dans les modules voulus
target_modules = []
for name, module in model.named_modules():
    if name.endswith('self_attn'):
        target_modules.append(name)
        
model_ = inject_linear_attention(
    model,
    model.config,
    target_modules=target_modules, #["model.layers.0.self_attn", "model.layers.1.self_attn"],
    operator_mode="delta_rule",
    fla_weight=0.5,
    chunk_size=64,
)

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    output = model_.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False  # déterministe
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

```

```python

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from liza.injector import AdjustMaGWeightCallback

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model_, peft_config)
#model.print_trainable_parameters()

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="tensorboard",
)

dataset = load_dataset("yahma/alpaca-cleaned")["train"].select(range(1000))

def format_instruction(sample):
    return {
        "text": f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
    }
dataset = dataset.map(format_instruction)

def tokenize(sample):
    tokens = tokenizer(
        sample["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
    callbacks=[AdjustMaGWeightCallback(model, initial_weight=0.01, final_weight=0.5, transition_step=500)]
)

trainer.train()

model.save_pretrained("./liza_llama-instruct_model")
tokenizer.save_pretrained("./iza_llama-instruct_model")
```


---

## Testing

Run the test suite with:

```bash
make test
```


---

## Development

- Code is organized into modular components under the `liza/` directory.
- Use `pytest` for testing and `pydantic` for configuration validation.
- Contributions and feature requests are welcome!

---

## Requirements

- Python 3.8+
- PyTorch
- einops
- pydantic
- flash-linear-attention

See `requirements.txt` for the full list.

---

## Contact

For questions or support, please open an issue on the GitHub repository or contact the maintainer.

---

This README provides a concise overview, installation instructions, usage example, and development notes to help users get started quickly with LiZA. Let me know if you want me to generate a `CHANGELOG.md` or examples folder as well!

