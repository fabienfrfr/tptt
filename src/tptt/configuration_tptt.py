from typing import Optional

from transformers import PretrainedConfig


class TpttConfig(PretrainedConfig):
    """
    Configuration class for the TPTT model.
    Compatible with HuggingFace's from_pretrained and save_pretrained methods.
    """

    model_type = "tptt"
    auto_map = {
        "AutoModelForCausalLM": "modeling_tptt.TpttModel",
        "AutoConfig": "configuration_tptt.TpttConfig",
    }

    def __init__(
        self,
        base_model_name: str = "gpt2",
        target_modules_names: Optional[list[str]] = ["attn", "self_attn", "attention"],
        operator_mode: str = "delta_rule",
        max_self_attn_length: Optional[int] = 2048,
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
        **kwargs,
    ):
        """
        Initialize TpttConfig with model and attention parameters.
        """

        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.target_modules_names = target_modules_names
        self.operator_mode = operator_mode
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size
        self.max_self_attn_length = max_self_attn_length


TpttConfig.register_for_auto_class()
