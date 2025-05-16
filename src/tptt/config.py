"""This script demonstrates how to use LiZA for linear attention injection."""


class TpttConfig:
    """Configuration for TPTT model."""

    def __init__(
        self,
        model_name: str,
        target_modules_names: str = "self_attn",
        operator_mode: str = "delta_rule",
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
    ):
        """Initialize the TPTT configuration."""
        self.model_name = model_name
        self.target_modules_names = target_modules_names
        self.operator_mode = operator_mode
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size
