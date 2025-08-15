# Titanesque Scripts

This Python script downloads and extracts **only the `scripts` folder** from the `main` branch of the [`fabienfrfr/tptt`](https://github.com/fabienfrfr/tptt) GitHub repository.

## Features

- Downloads the `.zip` archive of the `main` branch  
- Extracts its contents into a temporary folder  
- Moves only the `scripts` directory to the current working directory  
- Deletes temporary files after extraction  

## Python Code

```python
import requests
import zipfile
import io
import shutil

# URL to the ZIP file of the 'main' branch
repo_url = "https://github.com/fabienfrfr/tptt/archive/refs/heads/main.zip"

# Download the ZIP archive
resp = requests.get(repo_url)

# Open ZIP file from memory
z = zipfile.ZipFile(io.BytesIO(resp.content))

# Extract into a temporary folder
z.extractall("temp_repo")

# Move the 'scripts' directory to the current folder
shutil.move("temp_repo/tptt-main/scripts", "./scripts")

# Remove the temporary folder
shutil.rmtree("temp_repo")
```

## Launch Code

```sh
# Run LLaMA base with delta_rule, LoRA active, mag_weight 0.5
python train.py train \
  --model_name "meta-llama/Llama-3.2-1B" \
  --method delta_rule \
  --mag_weight 0.5

# Run Qwen with delta_product, no LoRA and no linear_attn
python train.py train \
  --model_name "Qwen/Qwen2.5-1.5B" \
  --method delta_product \
  --mag_weight 0.25 \
  --disable_linear_attn True \
  --lora False
```


## Notes

- If the `scripts` folder already exists, `shutil.move` will raise an error.  
  → Add a check or overwrite logic if needed.  
- This script requires only the `requests` package in addition to Python’s built-in modules.