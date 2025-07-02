# TPTT-LiZA Training Documentation

## Introduction

This doc is designed to train a language model based on the TPTT architecture using the Hugging Face library. It employs the LoRA (Low-Rank Adaptation) technique for efficient fine-tuning of large language models. TPTT enhances pretrained Transformer models with efficient linearized attention mechanisms (LiZA) and advanced memory management, notably the Memory as Gate (MaG) mechanism, to improve efficiency and scalability for long-context inference.


## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Model Configuration](#model-configuration)
3. [Model Titanization](#model-titanization)
4. [Data Loading and Preparation](#data-loading-and-preparation)
5. [Model Training](#model-training)
6. [Inference](#inference)


## Initial Setup

### Installing Dependencies

Begins by installing essential dependencies for quantization and TPTT model support:

```python

!pip install -q bitsandbytes accelerate
# !pip install -q git+https://github.com/fabienfrfr/tptt@dev
!pip install -q tptt

```

### Authentication

To access models and datasets from the Hugging Face Hub, you need to authenticate using your Hugging Face token. This is done using the huggingface_hub library.

```python


from huggingface_hub import login, HfApi
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
api = HfApi()

```


### Importing Libraries

Key libraries for model building, training, and data handling are imported:

```python

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import tptt

```

## Model Configuration

### Model Parameters

Define model type, base model name, and LoRA target modules:

```python

MODEL_TYPE = 'llama'
base_model_name = "meta-llama/Llama-3.2-1B"
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

```

### LoRA Configuration

LoRA is configured for parameter-efficient fine-tuning by injecting low-rank matrices into selected projection layers, reducing trainable parameters and memory usage while maintaining performance.

```python

lora_config = LoraConfig(
r=8,
lora_alpha=16,
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM",
target_modules=target_modules,
).to_dict()

```

## Model Titanization


Transforming a pretrained causal langage transformer model into Titans (Tptt)

```python

base_model_name="meta-llama/Llama-3.2-1B"
config = tptt.TpttConfig(
    base_model_name=base_model_name,
    lora_config=lora_config,
)
model = tptt.TpttModel(
    config, 
    attn_implementation="eager",
    # torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,
)

model.backbone.print_trainable_parameters()
```

After that, it's classical training from Transformer library. But `AdjustMaGWeightCallback` it's recommanded.

## Data Loading and Preparation

### Loading the Dataset

A dataset is loaded from Hugging Face and a subset is selected for training:

```python

raw_dataset = load_dataset(DATASET)["train"].select(range(N))

```

### Tokenization

The data is tokenized using a tokenizer compatible with the base model:

```python
def preprocess_fn(samples):

(...)
    tokens = tokenizer(
        prompts,
        truncation=True,
        max_length=384, #256, 512
        padding="longest",
        return_attention_mask=True,
    )
(...)
```

```python

tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
tokenized_dataset = raw_dataset.map(preprocess_fn, batched=True, remove_columns=raw_dataset.column_names)

```

## Model Training


```python
# LiZA MaG callback
initial_weight=0.01,
final_weight=0.5,
transition_step=100,
liza_callback = tptt.AdjustMaGWeightCallback(
            model,
            initial_weight=initial_weight,
            final_weight=final_weight,
            transition_step=transition_step,)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

### Training Configuration

Set up training arguments such as epochs, batch size, and learning rate. Mixed precision training and gradient clipping are used for efficiency and stability.

```python

training_args = TrainingArguments(
output_dir=dir_path,
per_device_train_batch_size=3,
num_train_epochs=EPOCH,
learning_rate=5e-4,
bf16=True,
seed=42,
)

```

### Initializing the Trainer

A Hugging Face `Trainer` is initialized with the model, training arguments, and tokenized dataset:

```python

trainer = Trainer(
model=model,
args=training_args,
train_dataset=tokenized_dataset,
data_collator=data_collator,
callbacks=[liza_callback],
)

```

### Launching Training

Start the training process:

```python

trainer.train()

```

#### Dynamic Memory as Gate Scheduling (LiZA MaG Callback)

During training, the MaG (Memory as Gate) parameter is dynamically adjusted. It starts at 0.01 and linearly increases to 0.5 over the first 100 steps, balancing the use of vanilla and linearized attention for optimal performance. This is integrated directly into the training loop for adaptive control.

## Saving and Loading the Model

### Saving the Model

After training, save the model and tokenizer:

```python

tokenizer.save_pretrained(dir_path)
model.save_pretrained(dir_path)

```

### Loading the Model

Reload the model for inference or further use:

```python

model_tptt = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)

```

## Inference

### Inference Pipeline

Use the TPTT pipeline to generate text from the trained model:

```python

pipe = tptt.TpttPipeline(model=model_tptt, tokenizer=tokenizer, device=device)
result = pipe("Bonjour, I'm Fabien Furfaro,", max_new_tokens=100)

```

## Conclusion

This docs demonstrates how to train a language model using the TPTT architecture and the LoRA technique. It covers all steps from installing dependencies to inference, including model configuration, data loading, dynamic attention scheduling, and training. TPTT’s integration of linearized attention and advanced memory management enables efficient and scalable adaptation of large language models using Hugging Face Transformers.