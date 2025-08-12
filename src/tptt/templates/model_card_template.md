---
language: en
license: apache-2.0
library_name: transformers
tags:
  - tptt
  - peft
  - trust_remote_code
pipeline_tag: text-generation
base_model: {{config.base_model_name}}
datasets:
- {{dataset}}
---

# {{model_id}}

<p align="center">
    <a href="https://arxiv.org/abs/2506.17671">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-tptt-blueviolet.svg">
    </a>
    <a href="https://pypi.org/project/tptt/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/tptt?color=orange">
    </a>
    <a href="https://github.com/fabienfrfr/tptt/">
        <img alt="Release" src="https://img.shields.io/github/v/release/fabienfrfr/tptt?color=brightgreen">
    </a>
    <a href="https://fabienfrfr.github.io/tptt/">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-online-blue">
    </a>
    <a href="https://huggingface.co/ffurfaro">
        <img alt="HuggingFace" src="https://img.shields.io/badge/hf-ffurfaro-yellow">
    </a>
</p>

Titanesque version of `{{config.base_model_name}}` with parallel linearized attention (TPTT 😊) and PEFT.

The architecture was presented in the paper [TPTT](https://huggingface.co/papers/2506.17671).


## Model Details

- **Architecture:** {{config.architectures}}
- **Base model:** {{config.base_model_name}}
- **LiZA config:** operator={{config.operator_mode}}, mag={{config.mag_weight}}
- **LoRA config:** r={{config.lora_config.r | default("")}}, alpha={{config.lora_config.lora_alpha | default("")}}, dropout={{config.lora_config.lora_dropout | default("")}}
- **torch_dtype:** {{torch_dtype}}

## Usage


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
"ffurfaro/{{model_id}}",
trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("ffurfaro/{{config.base_model_name}}")

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs, skip_special_tokens=True))

```

> [!IMPORTANT]
> You must specify the `subfolder` if the repo contains multiple models, see the homepage for details.

## Training

- **Dataset:** {{dataset}}
- **Platform:** {{platform}}
- **Hardware:** {{hardware}}
- **Batch size:** {{batch_size}}
- **Epochs:** {{epochs}}
- **Learning rate (final):** {{learning_rate}}
- **Loss (final):** {{loss}}
- **Training runtime:** {{train_runtime}} sec
- **Samples per second:** {{train_samples_per_second}}
- **Steps per second:** {{train_steps_per_second}}
- **Total FLOPs:** {{total_flos}}
- **Gradient norm (final):** {{grad_norm}}

## Evaluation

- **Metrics:** Training loss only (no eval yet, table soon : PiQA, ARC, Hella, Wino, GSM8K, MMLU)
- **Results:** Final training loss: {{loss}}


## Citation & Contact

If you use TPTT in your academic work, please cite [Furfaro](https://huggingface.co/ffurfaro). For questions or support, please open an issue on the [GitHub repository](https://github.com/fabienfrfr/tptt) or contact the maintainer.


---