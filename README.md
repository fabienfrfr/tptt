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

```python
def main(training=True):
    # 1. Transforming Pretrained Transformer into Memory as Gate (Inject LiZA by default)
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tptt_model = TpttModel(model_name)

    if training :
        # 2. Inject LoRA parameters into the model
        tptt_model.inject_lora_parameters()

        # 3. Load a small subset of the dataset for training
        from datasets import load_dataset
        dataset = load_dataset("yahma/alpaca-cleaned")["train"].select(range(100))  # 100 samples for quick testing

        # 4. Train the model
        tptt_model.train(dataset=dataset)

    # 5. Generate text with the trained model
    prompt = "Once upon a time,"
    generated_text = tptt_model.generate(prompt)
    print("Generated text:", generated_text)

    # 6. Save the model and tokenizer
    tptt_model.save_model("./liza_llama-instruct_model")

if __name__ == "__main__":
    main()

```


---

## Testing

Run the test suite with:

```bash
make test
```


---

## Development

- Code is organized into modular components under the `src/tptt` directory.
- Use `pytest` for testing.
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
@misc{furfaro2025liza,
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
