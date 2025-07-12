"""
Author : Fabien FURFARO
"""

import os
import re
from typing import List, Optional, Union

import torch
from transformers import AutoConfig, PretrainedConfig


def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_sets_to_lists(x) for x in obj]
    else:
        return obj


class TpttConfig(PretrainedConfig):
    """
    Configuration class for the TPTT model.
    This class merges the backbone config (e.g., Llama) with custom TPTT parameters,
    """

    model_type = "tptt"
    auto_map = {
        "AutoModelForCausalLM": "modeling_tptt.TpttModel",
        "AutoConfig": "configuration_tptt.TpttConfig",
    }
    architectures = ["TpttModel"]

    def __init__(
        self,
        base_model_config: Optional[Union[dict, PretrainedConfig]] = None,
        base_model_name: str = "meta-llama/Llama-3.2-1B",
        name_or_path: Optional[str] = None,
        target_modules_names: Optional[List[str]] = None,
        force_attn_implementation: Optional[str] = "eager",
        operator_mode: str = "delta_rule",
        max_self_attn_length: Optional[
            int
        ] = None,  # unnecessary if SWA, else, standards 8192
        base_scale_attn: bool = False,
        mag_weight: float = 0.5,  # if 1.0, use only linear operator
        cross_gate: bool = False,  # unlinear mixing strategy
        max_chunk_size: int = 64,
        linear_precision: Union[str, torch.dtype] = "float32",
        lora_config: Optional[dict] = None,  # only serialized accepted
        **kwargs,
    ):
        # If base_model_config is provided, load it and merge with this config
        if base_model_config is not None:
            if isinstance(base_model_config, PretrainedConfig):
                base_model_config = base_model_config.to_dict()
        else:
            # Load config from Hugging Face Hub or a local path
            base_model_config = AutoConfig.from_pretrained(
                base_model_name, **kwargs
            ).to_dict()
        # Merge all backbone fields into this config
        for k, v in base_model_config.items():
            setattr(self, k, v)

        self.base_model_name = base_model_name
        self._name_or_path = (
            name_or_path
            if name_or_path is not None
            else "Titans-" + base_model_name.split("/", 1)[1]
        )

        self.target_modules_names = target_modules_names or [
            "attn",
            "self_attn",
            "attention",
        ]
        self.force_attn_implementation = force_attn_implementation
        self.operator_mode = operator_mode
        self.base_scale_attn = base_scale_attn
        self.mag_weight = mag_weight
        self.cross_gate = cross_gate
        self.max_chunk_size = max_chunk_size
        self.max_self_attn_length = max_self_attn_length

        if isinstance(linear_precision, torch.dtype):
            linear_precision = str(linear_precision).replace("torch.", "")
        self.linear_precision = linear_precision

        self.lora_config = lora_config
        if lora_config is not None:
            if hasattr(self.lora_config.get("peft_type"), "value"):
                self.lora_config["peft_type"] = self.lora_config["peft_type"].value
            self.lora_config = convert_sets_to_lists(self.lora_config)

        super().__init__(**kwargs)  # flush unconsistend pretrained parameters (?)
        # Copy class attributes to instance for serialization (save dict)
        self.model_type = self.__class__.model_type
        self.auto_map = self.__class__.auto_map
        self.architectures = self.__class__.architectures


TpttConfig.register_for_auto_class()


def extract_template_variables(template):
    return set(re.findall(r"\{([^{}]+)\}", template))


def generate_model_card(path: str, config, **kwargs):
    """Generate model card from template and training metadata."""
    template_path = os.path.join(os.path.dirname(__file__), "model_card_template.md")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Flatten config
    def flatten_config(config):
        result = {}
        if hasattr(config, "__dict__"):
            config = config.__dict__
        for k, v in config.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    result[f"{k}_{subk}"] = subv
            else:
                result[k] = v
        return result

    variables = flatten_config(config)
    variables.update(kwargs)
    variables["model_id"] = os.path.basename(path)

    # Extract variables from template
    template_vars = extract_template_variables(template)

    # Add default values for missing variables
    for var in template_vars:
        if var not in variables:
            variables[var] = "N/A"

    # Handle list conversion (optional but useful)
    for k, v in variables.items():
        if isinstance(v, list):
            variables[k] = ", ".join(map(str, v))

    model_card_content = template.format(**variables)
    with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card_content)
