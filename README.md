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
import tptt
from datasets import load_dataset

def main(training=True):
    # 1. Config et modèle
    config = tptt.TpttConfig(base_model_name="gpt2")
    model = tptt.TpttModel(config)

    # 2. (Optionnel) Injection LoRA
    model.add_lora()

    # 3. Préparation du dataset
    dataset = load_dataset("yahma/alpaca-cleaned")["train"].select(range(100))
    # (instruction_format est importé via tptt)
    dataset = dataset.map(tptt.instruction_format)

    # 4. Entraînement
    trainer = tptt.TpttTrainer(model, dataset)
    trainer.train()

    # 5. Génération
    pipe = tptt.TpttPipeline(model)
    print(pipe("Once upon a time,"))

    # 6. Sauvegarde
    model.save_pretrained("./my_tptt_model")

if __name__ == "__main__":
    main()

```

---

## Development

- Code is organized into modular components under the `src/tptt` directory.
- Use `pytest` for testing and `sphinx` for documentation.
- Contributions and feature requests are welcome!

---

## Requirements

- Python 3.12+
- PyTorch
- einops
- Transformers
- flash-linear-attention

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
