# LiZA: LineariZe Attention Injection

LiZA is a modular Python library designed to inject efficient linearized attention mechanisms-such as Gated Linear Attention (GLA) and Delta Rule Attention-into existing Transformer models. It leverages the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library for high-performance implementations, enabling scalable and memory-efficient attention computations.

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

