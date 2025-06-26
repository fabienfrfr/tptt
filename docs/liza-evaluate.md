# TPTT_LiZA_Evaluation Documentation

This document provides an overview of the TPTT_LiZA_Evaluation, which is designed to evaluate a language model using the LightEval framework and Hugging Face Transformers library.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Authentication](#authentication)
3. [Model Configuration](#model-configuration)
4. [Model Loading and Inference](#model-loading-and-inference)
5. [Evaluation with LightEval](#evaluation-with-lighteval)
6. [Running the Evaluation Pipeline](#running-the-evaluation-pipeline)

## Setup and Installation

Begins by installing necessary packages and dependencies. This includes packages for handling language models, evaluation, and other utilities.

```python

!pip install -q -U git+https://github.com/fabienfrfr/tptt@dev
!pip install -q -U lighteval

```

## Authentication

To access models and datasets from the Hugging Face Hub, you need to authenticate using your Hugging Face token. This is done using the huggingface_hub library.

```python


from huggingface_hub import login, HfApi
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
api = HfApi()

```

## Model Configuration

The base model is specified, and its state dictionary is loaded to inspect the model's layers and parameters.

```python


base_model_name = "ffurfaro/Titans-Llama-3.2-1B"

from huggingface_hub import hf_hub_download
repo_id = base_model_name
filename = "adapter_model.safetensors"
local_path = hf_hub_download(repo_id=repo_id, filename=filename)

from safetensors.torch import load_file
state_dict = load_file(local_path)
for key in state_dict.keys():
print(key)

```

## Model Loading and Inference

The model and tokenizer are loaded using the transformers library. An example prompt is tokenized and passed through the model to generate a response.

```python


from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

prompt = "Bonjour, I'm Fabien Furfaro, "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs))

```

## Evaluation with LightEval

LightEval is used to evaluate the model on various benchmarks. The evaluation tracker, pipeline parameters, and model configuration are set up.

```python

import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
accelerator = None

evaluation_tracker = EvaluationTracker(
output_dir="./results",
save_details=True,
)

pipeline_params = PipelineParameters(
launcher_type=ParallelismManager.ACCELERATE,
custom_tasks_directory=None,
max_samples=10
)

model_config = TransformersModelConfig(
tie_word_embeddings=True,
model_name=base_model_name,
trust_remote_code=True,
dtype="bfloat16",
use_chat_template=True,
)

```

## Running the Evaluation Pipeline

The evaluation pipeline is created and run to evaluate the model on specified tasks. Results are saved and displayed.

```python


task = "helm|mmlu|1|1"

pipeline = Pipeline(
tasks=task,
pipeline_parameters=pipeline_params,
evaluation_tracker=evaluation_tracker,
model_config=model_config,
)

pipeline.evaluate()
pipeline.save_and_push_results()
pipeline.show_results()

```

## Conclusion

This notebook provides a comprehensive guide to evaluating a language model using Hugging Face Transformers and LightEval. It covers everything from setting up the environment to running evaluations and interpreting results.