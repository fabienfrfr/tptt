<h1 align="center"> <p>😊 TPTT</p></h1>

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

<h3 align="center">
    <p>Transforming Pretrained Transformers into Titans </p>
</h3>


**TPTT** is a modular Python library designed to inject efficient linearized attention (*LiZA*) mechanisms-such as *Memory as Gate* (described in [Titans](https://arxiv.org/abs/2501.00663))-into pretrained transformers 🤗.


---

## Features

- **Flexible Attention Injection**: Seamlessly wrap and augment standard Transformer attention layers with linearized attention variants for latent memory.
- **Support for Linear Attention**: Includes implementations of [DeltaNet](https://arxiv.org/abs/2406.06484) and [DeltaProduct](https://arxiv.org/abs/2502.10297) with optional recurrent nonlinearity between chunks.
- **Modular Design**: Easily extend or customize operators and integration strategies.
- **Compatibility**: Designed to integrate with Hugging Face Transformers and similar PyTorch models.

> [!IMPORTANT]
> After the injection of LiZA, the model must be fine‑tuned for alignment and obtain memory.

![overview](./docs/fig.png)

> **Note**: Order 2 `Delta-Product` has the same expressiveness as titans.


## Installation and Usage

```bash
pip install tptt
```

#### *Titanesque Documentation*

- [TPTT-LiZA_Training](./docs/liza-training.md):  
  Instructions for training TPTT-based models with LoRA and advanced memory management.

- [TPTT_LiZA_Evaluation](./docs/liza-evaluate.md):  
  Guide for evaluating language models with LightEval and Hugging Face Transformers.

- [TPTT_LiZA_FromScratch](./docs/liza-from-scratch.md):  
  Integrating the `LinearAttention` module into Pytorch deep learning projects.

Basic usage :

```python

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import tptt
from tptt import save_tptt_safetensors, get_tptt_model, load_tptt_safetensors
from torch import nn

##### Transforming into Titans (Tptt)
base_model_path = "Qwen/Qwen2.5-1.5B"
base_config = AutoConfig.from_pretrained(base_model_path)
base_model_name = "Qwen/Qwen2.5-1.5B"
tptt_config = tptt.TpttConfig(
    base_model_config=base_config,
    base_model_name= base_model_name, 
    #lora_config=lora_config,

)
model = tptt.TpttModel(config)
# manual local save
save_tptt_safetensors(model, path, name)

##### Pretrained Titans from Transformer
repo_id = "ffurfaro/Titans-Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

##### More custom for other Model (BERT, ViT, etc.)
model, linear_cache = get_tptt_model(model, config) # you can activate Bidirectional
model = load_tptt_safetensors(repo_or_path, model) # from saved LoRA only

##### Using LinearAttention from scratch
layers = nn.ModuleList([
    tptt.LinearAttention(hidden_dim=64, num_heads=4,)
    for _ in range(num_layers)])

```

Some `scripts` are available [here](./scripts/)  

---

## Development

- Code is organized into modular components under the `src/tptt` directory.
- Use `pytest` for testing and `sphinx` for documentation. See on this [link](https://fabienfrfr.github.io/tptt/)🔥
- Contributions and feature requests are welcome!

---

## Requirements

- Python 3.11+
- PyTorch
- einops
- Transformers
- Peft

See `requirements.txt` for the full list.

---

## Citation

If you use TPTT in your academic work, please cite:

```bibtex
@article{furfaro2025tptt,
  title={TPTT: Transforming Pretrained Transformer into Titans},
  author={Furfaro, Fabien},
  journal={arXiv preprint arXiv:2506.17671},
  year={2025}
}
```


---

## Contact

For questions or support, please open an issue on the [GitHub repository](https://github.com/fabienfrfr/tptt) or contact the maintainer.
