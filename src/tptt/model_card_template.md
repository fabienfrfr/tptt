---
language: en
license: apache-2.0
library_name: transformers
tags:
  - tptt
  - peft
  - trust_remote_code
pipeline_tag: text-generation
base_model: {base_model_name}
datasets:
- {dataset}
---

# {model_id}

Titanesque version of `{base_model_name}` with parallel linearized attention (TPTT 😊) and PEFT.

The architecture was presented in the paper [TPTT](https://huggingface.co/papers/2506.17671).


## Model Details

- **Architecture:** {architectures}
- **Base model:** {base_model_name}
- **LiZA config:** operator={operator_mode}, mag={mag_weight}
- **LoRA config:** r={lora_config_r}, alpha={lora_config_lora_alpha}, dropout={lora_config_lora_dropout}
- **torch_dtype:** {torch_dtype}

## Usage


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
"ffurfaro/{model_id}",
trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("ffurfaro/{model_id}")

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs, skip_special_tokens=True))

```

## Training

- **Dataset:** {dataset}
- **Platform:** {platform}
- **Hardware:** {hardware}
- **Batch size:** {batch_size}
- **Epochs:** {epochs}
- **Learning rate (final):** {learning_rate}
- **Loss (final):** {loss}
- **Training runtime:** {train_runtime} sec
- **Samples per second:** {train_samples_per_second}
- **Steps per second:** {train_steps_per_second}
- **Total FLOPs:** {total_flos}
- **Gradient norm (final):** {grad_norm}

## Evaluation

- **Metrics:** Training loss only (no eval yet, table soon : PiQA, ARC, Hella, Wino, GSM8K, MMLU)
- **Results:** Final training loss: {loss}


## Citation & Contact

If you use TPTT in your academic work, please cite [Furfaro](https://huggingface.co/ffurfaro). For questions or support, please open an issue on the [GitHub repository](https://github.com/fabienfrfr/tptt) or contact the maintainer.


---