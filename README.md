# LiZA: LineariZe Attention Injection

LiZA is a modular Python library designed to inject efficient linearized attention mechanisms-such as Gated Linear Attention (GLA) and Delta Rule Attention-into existing Transformer models. It leverages the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) library for high-performance implementations, enabling scalable and memory-efficient attention computations.

---

## Features

- **Flexible Attention Injection**: Seamlessly wrap and augment standard Transformer attention layers with linearized attention variants.
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
from transformers import AutoModelForCausalLM
from liza import CustomInjectConfig, get_custom_injected_model

# Load your pretrained model (example)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define injection configuration
target_modules = []
for name, module in model.named_modules():
    # Vérifiez que le nom du module se termine par 'self_attn'
    if name.endswith('self_attn'):
        target_modules.append(name)

config = CustomInjectConfig(
    target_modules=target_modules, #["transformer.h.0.attn"],  # Example module name(s)
    operator="gla",                          # or "delta_rule"
    fla_weight=0.1                          # Weight for linearized attention output
)

# Inject linearized attention
model = get_custom_injected_model(model, config)
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

