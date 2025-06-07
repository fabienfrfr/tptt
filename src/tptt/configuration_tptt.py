from typing import List, Optional, Union
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
        base_model_config: Optional[Union[str, dict, PretrainedConfig]] = None,
        base_model_name: str = "meta-llama/Llama-3.2-1B",
        name_or_path: Optional[str] = None,
        target_modules_names: Optional[List[str]] = None,
        operator_mode: str = "delta_rule",
        max_self_attn_length: int = 4096,
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
        lora_config: Optional[dict] = None,  # only serialized accepted
        **kwargs,
    ):

        if base_model_config is not None:
            if isinstance(base_model_config, str):
                # Load config from Hugging Face Hub or a local path
                base_model_config = AutoConfig.from_pretrained(
                    base_model_config
                ).to_dict()
            elif isinstance(base_model_config, PretrainedConfig):
                base_model_config = base_model_config.to_dict()
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
        self.operator_mode = operator_mode
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size
        self.max_self_attn_length = max_self_attn_length

        self.lora_config = lora_config
        if lora_config is not None:
            if hasattr(self.lora_config.get("peft_type"), "value"):
                self.lora_config["peft_type"] = self.lora_config["peft_type"].value
            self.lora_config = convert_sets_to_lists(self.lora_config)

        super().__init__(**kwargs)
        # Copy class attributes to instance for serialization (save dict)
        self.model_type = self.__class__.model_type
        self.auto_map = self.__class__.auto_map
        self.architectures = self.__class__.architectures


TpttConfig.register_for_auto_class()
