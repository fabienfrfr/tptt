<h1 align="center"> <p>😊 TPTT</p></h1>
<h3 align="center">
    <p>Transforming Pretrained Transformers into Titans (TPTT) </p>
</h3>

**TPTT** is a modular Python library designed to inject efficient linearized attention (*LiZA*) mechanisms-such as *Memory as Gate* (described in [Titans](https://arxiv.org/html/2501.00663v1))-into pretrained transformers 🤗.
It leverages the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library for high-performance implementations, enabling scalable and memory-efficient attention computations. (in progress 🔥)

---

## Features

- **Flexible Attention Injection**: Seamlessly wrap and augment standard Transformer attention layers with linearized attention variants for latent memory.
- **Support for GLA and Delta Rule**: Includes implementations of Gated Linear Attention and Delta Rule Attention.
- **Modular Design**: Easily extend or customize operators and integration strategies.
- **Compatibility**: Designed to integrate with Hugging Face Transformers and similar PyTorch models.


![overview](./docs/fig.png)


---

## Installation

```bash
git clone https://github.com/fabienfrfr/tptt.git
cd tptt
make install
```

> **Note**: `flash-linear-attention` requires a CUDA-enabled GPU.

---

## Complete Usage Example

```bash
#!pip install -q flash-linear-attention
!pip install -q bitsandbytes accelerate
!pip install -q -U git+https://github.com/fabienfrfr/tptt@main # PyPi soon
```

#### *Titanesque* Import


```python

from transformer import AutoTokenizer, AutoModelForCausalLM
import tptt

repo_id = "ffurfaro/Titans-Llama-3.2-1B"

# Import model and tokenizer
model_tptt = AutoModelForCausalLM.from_pretrained(repo_id, token=hf_token, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Prepare for inference
device = 0 if torch.cuda.is_available() else -1
model_tptt.to(f"cuda:{device}" if device != -1 else "cpu")

model_tptt.eval()
pipe = tptt.TpttPipeline(model=model_tptt, tokenizer=tokenizer, device=device)

# Generate text
result = pipe("Bonjour, I'm Fabien Furfaro,", max_new_tokens=100)
print(result[0]["generated_text"])

```

#### *Titanesque* Training


```python

from transformer import AutoTokenizer, AutoModelForCausalLM
import tptt

base_model_name="meta-llama/Llama-3.2-1B"


##### Peft parameters

target_modules = ["q_proj","k_proj","v_proj","o_proj"]  # Llama, Mistral, OLMo. Minimal : q_proj, v_proj

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
).to_dict()

##### Transforming into Titans (Tptt)
config = tptt.TpttConfig(
    base_model_name=base_model_name,
    lora_config=lora_config,
)

model = tptt.TpttModel(config, backbone=backbone) #     torch_dtype=torch.bfloat16, # for model trained in float32 
model.backbone.print_trainable_parameters()

##### Preprocessing

tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, token=hf_token)
# Ensure the tokenizer has a padding token for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

raw_dataset = load_dataset("yahma/alpaca-cleaned")["train"].select(range(N))

def preprocess_fn(samples):
    """
    Tokenize the samples for causal language modeling.
    Concatenate instruction, input, and output as needed.
    """
    prompts = [
        f"{instr}\n{inp}" if inp else instr
        for instr, inp in zip(samples["instruction"], samples["input"])
    ]
    # Optionally, append output for supervised fine-tuning
    prompts = [f"{p}\n{out}" for p, out in zip(prompts, samples["output"])]
    tokens = tokenizer(
        prompts,
        truncation=True,
        max_length=512, #256,
        padding="longest", #padding= "max_length",
        return_attention_mask=True,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = raw_dataset.map(
    preprocess_fn, batched=True, remove_columns=raw_dataset.column_names
)

# Tokenize the dataset in batches and remove original columns
tokenized_dataset = raw_dataset.map(
    preprocess_fn, batched=True, remove_columns=raw_dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define HuggingFace TrainingArguments for reproducible training
training_args = TrainingArguments(
    output_dir="./tptt_output",
    per_device_train_batch_size=4, # per_device_train_batch_size * N GPU --> VRAM limit risk 
    num_train_epochs=EPOCH,
    learning_rate=  5e-4,
    max_grad_norm=1.0, # gradiant clipping
    bf16=True,  # Use mixed precision if supported by hardware (underflow gradient)
    ddp_find_unused_parameters=False, 
    logging_steps=5,
    save_total_limit=2,  # Limit HDD
    seed=42,
    save_strategy="epoch",
    report_to="tensorboard",
)

# LiZA MaG callback
initial_weight=0.01,
final_weight=0.5,
transition_step=100,
liza_callback = tptt.AdjustMaGWeightCallback(
            model,
            initial_weight=initial_weight,
            final_weight=final_weight,
            transition_step=transition_step,)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[liza_callback],
)

trainer.train()

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
- Peft (optional)
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
