# Titanesque Scripts

This Python script downloads and extracts **`scripts`** from the `main` branch of the [fabienfrfr/tptt](https://github.com/fabienfrfr/tptt) GitHub repository.

⚠️ This folder contains prototype scripts for experimentation and testing purposes only. It is not designed or tested for production use. This scripts is designed if you want to reproduce the results in [TPTT](https://arxiv.org/abs/2506.17671) paper.


## Usage on Notebook

### 1. Clone the repository

Run this cell in your Kaggle notebook:

```bash
!git clone --depth 1 --branch main https://github.com/fabienfrfr/tptt.git
```

***

### 2. Prepare environment

```python
import os
QUANTIZED=False
if QUANTIZED :
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["MODEL_NAME"] = "meta-llama/Llama-3.2-1B" 
os.environ["QUANTIZATION"] = "" if QUANTIZED else "--no-model-quantized"
```

### 3. Launch training commands

Run training using your desired model and method. For test example:

```bash
!python tptt/scripts/train.py --model-name $MODEL_NAME --dataset-percentage 0.001 --max-length 128 $QUANTIZATION --token $HF_TOKEN
```


***

⚠️ Disclaimer: Experimental (in progress) ⚠️

## Training Commands to Cover All Experiment Variants

### 1. Base training variants (all models)

```sh
# delta_rule with mag_weight=0.5 and LoRA enabled
python train.py train --model_name MODEL_NAME --method delta_rule --mag_weight 0.5 --lora True

# delta_product with mag_weight=0.5 and LoRA enabled
python train.py train --model_name MODEL_NAME --method delta_product --mag_weight 0.5 --lora True
```

Replace `MODEL_NAME` with one of:  
`meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-1.5B`, `apple/OpenELM-1_1B`, `mistralai/Mistral-7B-v0.3`, `allenai/OLMoE-1B-7B-0924`, `google/gemma-3-270m`, `Dream-org/Dream-v0-Base-7B`.

***

### 2. LLaMA-specific additional variants

```sh
# Without LoRA variants
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --mag_weight 0.5 --lora False
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --mag_weight 0.5 --lora False

# Cross Gate Mode experiments (mag_weight=0.5)
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --mag_weight 0.5 --cross_gate_mode True
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --mag_weight 0.5 --cross_gate_mode True

# Extra operator mode experiments (mag_weight=0.5)
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule_gelu --mag_weight 0.5
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product_r --mag_weight 0.5
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product_c --mag_weight 0.5
```

***

### 3. Liza Callback Sweeps (LLaMA only)

```sh
# Constant mag_weight values for delta_rule and delta_product
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --mag_weight 0.125
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --mag_weight 0.25
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --mag_weight 0.75

python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --mag_weight 0.125
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --mag_weight 0.25
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --mag_weight 0.75

# Transition schedules hitting mag_weight=0.5 over different steps
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --liza_mode gradual
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --liza_mode gradual --liza_transition_steps 10
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --liza_mode gradual --liza_transition_steps 1000

python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --liza_mode gradual
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --liza_mode gradual --liza_transition_steps 10
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --liza_mode gradual --liza_transition_steps 1000

# Alternating mag_weight schedules
python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_rule --liza_mode cyclic

python train.py train --model_name meta-llama/Llama-3.2-1B --method delta_product --liza_mode cyclic
```

***

## Notes

- The dataset used is 10% of `alpaca-cleaned` (~5000 samples), padded with `"longest"` padding and max length 386 tokens.  
- Seed is fixed to 42 for reproducibility.  
- The test set consists of 2 samples per run for sanity checking.  
- Batch size should be adjusted based on available GPU memory.  
- Support for quantization, linear attention disabling, cross gate mode, and rotation are available through command line flags.
