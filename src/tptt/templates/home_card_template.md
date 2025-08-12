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


## Model list

Classic model parameter with LiZA injection :

| Subfolder                      | Max Self Attn Length | Mag Weight | Cross Gate | Max Chunk Size | Bidirectional | LoRA | Description                                           |
|-------------------------------|----------------------|------------|------------|----------------|---------------|------|-------------------------------------------------------|
| delta_rule  | 8192 (default)       | 0.5        | False      | 64             | False         | Yes  | Parallel linearized attention with delta_rule operator|
| delta_rule_gelu | 8192 (default) | 0.5        | False      | 64             | False         | Yes  | Non-linear operator with gelu activation              |
| delta_product    | 8192 (default) | 0.5        | False      | 64             | False         | Yes  | Second order operator with derivative trick              |
| delta_product_r  | 8192 (default) | 0.5        | False      | 64             | False         | Yes  | Second order operator with rotative trick             |
| delta_product_c  | 8192 (default) | 0.5        | False      | 64             | False         | Yes  | Second order operator with combined trick             |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
"ffurfaro/{{model_id}}",
subfolder="tptt_subfolder", # see in repo tree
trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("ffurfaro/{{config.base_model_name}}")

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs, skip_special_tokens=True))

```


## Citation & Contact

If you use TPTT in your academic work, please cite [Furfaro](https://huggingface.co/ffurfaro). For questions or support, please open an issue on the [GitHub repository](https://github.com/fabienfrfr/tptt) or contact the maintainer.


---