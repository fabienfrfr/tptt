<h1 align="center"> <p>😊 TPTT</p></h1>
<h3 align="center">
    <p>Transforming Pretrained Transformers into Titans (TPTT) </p>
</h3>

**TPTT** is a modular Python library designed to inject efficient linearized attention (*LiZA*) mechanisms-such as *Memory as Gate* (described in [Titans](https://arxiv.org/html/2501.00663v1))-into pretrained transformers 🤗.
It leverages the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library for high-performance implementations, enabling scalable and memory-efficient attention computations.

---

## Features

- **Flexible Attention Injection**: Seamlessly wrap and augment standard Transformer attention layers with linearized attention variants for latent memory.
- **Support for GLA and Delta Rule**: Includes implementations of Gated Linear Attention and Delta Rule Attention.
- **Modular Design**: Easily extend or customize operators and integration strategies.
- **Compatibility**: Designed to integrate with Hugging Face Transformers and similar PyTorch models.

---

## Installation

```bash
git clone https://github.com/fabienfrfr/tptt.git
cd tptt
make install
```

> **Note**: `flash-linear-attention` requires a CUDA-enabled GPU and a compatible PyTorch version.

---

## Complete Usage Example

```bash
pip install -q -U git+https://github.com/fabienfrfr/tptt@main
```

```python
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import tptt

# 1. Configure and instantiate the TPTT model (with LoRA for efficient fine-tuning)
config = tptt.TpttConfig(base_model_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
model = tptt.TpttModel(config)
model.add_lora()

# 2. Prepare the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

# 3. Load and preprocess the dataset (Alpaca, 100 samples)
raw_dataset = load_dataset("yahma/alpaca-cleaned")["train"].select(range(100))
def preprocess_fn(samples):
    prompts = [
        f"{instr}\n{inp}" if inp else instr
        for instr, inp in zip(samples["instruction"], samples["input"])
    ]
    prompts = [f"{p}\n{out}" for p, out in zip(prompts, samples["output"])]
    tokens = tokenizer(prompts, truncation=True, max_length=256, padding="max_length", return_attention_mask=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
tokenized_dataset = raw_dataset.map(preprocess_fn, batched=True, remove_columns=raw_dataset.column_names)

# 4. Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./tptt_output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",  # disables TensorBoard for brevity
)

# 6. (Optional) MaG/Liza callback for dynamic weight adjustment
liza_callback = tptt.AdjustMaGWeightCallback(
    model, initial_weight=0.01, final_weight=0.5, transition_step=100
)

# 7. HuggingFace Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[liza_callback],
)

# 8. Train the model
trainer.train()

# 9. Prepare for inference
device = 0 if torch.cuda.is_available() else -1
model.to(f"cuda:{device}" if device != -1 else "cpu")#.eval()
pipe = tptt.TpttPipeline(model=model.backbone, tokenizer=tokenizer, device=device)

# 10. Generate text
result = pipe("Once upon a time,", max_new_tokens=100)
print(result[0]["generated_text"])

```

---

## Development

- Code is organized into modular components under the `src/tptt` directory.
- Use `pytest` for testing and `sphinx` for documentation.
- Contributions and feature requests are welcome!

---

## Requirements

- Python 3.11+
- PyTorch
- einops
- Transformers
- flash-linear-attention (optional)

See `requirements.txt` for the full list.

---

## Citation

If you use TPTT in your academic work, please cite:

```bibtex
@misc{furfaro2025tptt,
  author       = {Fabien Furfaro},
  title        = {TPTT: Transforming Pretrained Transformer into Titans},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/fabienfrfr/tptt}}
}
```


---

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/fabienfrfr/tptt) or contact the maintainer.
