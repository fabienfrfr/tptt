"""This module implements the TPTT model with linear attention and LoRA support."""

from typing import List, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, Pipeline, PretrainedConfig,
                          PreTrainedModel)

from .injection import inject_linear_attention
from .liza.memory_gate import LiZAttention
from .utils import Cache


class TpttConfig(PretrainedConfig):
    """
    Configuration class for the TPTT model.
    Compatible with HuggingFace's from_pretrained and save_pretrained methods.
    """

    model_type = "tptt"

    def __init__(
        self,
        base_model_name: str = "gpt2",
        base_tokenizer_name: Optional[str] = None,
        target_modules_names: Optional[List[str]] = None,
        operator_mode: str = "delta_rule",
        attn_ratio: float = 0.5,
        mag_weight: float = 0.5,
        max_chunk_size: int = 64,
        **kwargs,
    ):
        """
        Initialize TpttConfig with model and attention parameters.

        Args:
            base_model_name (str): Name or path of the base pretrained model.
            base_tokenizer_name (str, optional): Name or path of the tokenizer. If None, uses base_model_name.
            target_modules_names (list of str, optional): List of module name suffixes to target for attention injection.
            operator_mode (str): Operator mode for attention.
            mag_weight (float): Weight for MaG operator.
            max_chunk_size (int): Maximum chunk size for attention.
            **kwargs: Additional parameters for PretrainedConfig.
        """
        if target_modules_names is None:
            target_modules_names = ["attn", "self_attn", "attention"]
        if base_tokenizer_name is None:
            base_tokenizer_name = base_model_name

        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.base_tokenizer_name = base_tokenizer_name
        self.target_modules_names = target_modules_names
        self.operator_mode = operator_mode
        self.attn_ratio = attn_ratio
        self.mag_weight = mag_weight
        self.max_chunk_size = max_chunk_size


class TpttModel(PreTrainedModel):
    """
    TPTT model wrapper with linear attention (LiZA) and LoRA support.
    Only handles architecture and weights.
    """

    config_class = TpttConfig

    def __init__(self, config: TpttConfig):
        """
        Initialize TpttModel.

        Args:
            config (TpttConfig): Model configuration.
        """
        super().__init__(config)
        # Load the base pretrained model (e.g., Llama, Mistral, etc.)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            attn_implementation="eager",  # For LiZA compatibility
        )
        # Cache object for attention modules (if needed)
        self.cache = Cache()
        # Inject custom linear attention modules (LiZA)
        self._inject_liza_attention()

    def _inject_liza_attention(self):
        """
        Inject LiZAttention into the specified target modules of the base model.
        """
        # Find target modules by suffix (e.g., "attn", "attention")
        target_modules = [
            name
            for name, _ in self.backbone.named_modules()
            if any(name.endswith(suffix) for suffix in self.config.target_modules_names)
        ]
        if not target_modules:
            raise ValueError(
                f"Target modules '{self.config.target_modules_names}' not found in the model."
            )
        # Inject LiZAttention (external function, not shown here)
        self.backbone, self.cache = inject_linear_attention(
            self.backbone,
            self.backbone.config,
            liza_attention=LiZAttention,
            target_modules=target_modules,
            operator_mode=self.config.operator_mode,
            mag_weight=self.config.mag_weight,
            max_chunk_size=self.config.max_chunk_size,
        )

    def add_lora(self, lora_config: Optional[LoraConfig] = None):
        """
        Add LoRA adapters to the model.

        Args:
            lora_config (LoraConfig, optional): LoRA configuration. If None, use default and auto-detect target modules.
        """
        if lora_config is None:
            # Detect candidate modules for LoRA injection
            candidate_names = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # Llama, Mistral, OLMo
                "qkv_proj",
                "out_proj",  # OpenELM, some GPTs
                "c_attn",
                "c_proj",  # GPT-2
            ]
            target_modules = [
                name
                for name, _ in self.backbone.named_modules()
                if any(name.endswith(n) for n in candidate_names)
            ]
            target_modules = list(set(target_modules))
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
        # Inject LoRA adapters (external function, not shown here)
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass. All arguments are passed to the underlying base model.
        """
        return self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    def save_pretrained(self, path: str, **kwargs):
        """
        Save model weights and config to the given path.

        Args:
            path (str): Output directory.
        """
        self.backbone.save_pretrained(path, **kwargs)
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """
        Load model weights and config from the given path.

        Args:
            path (str): Directory with model/config files.

        Returns:
            TpttModel: Loaded model.
        """
        config = TpttConfig.from_pretrained(path)
        obj = cls(config)
        obj.backbone = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        return obj


class TpttPipeline(Pipeline):
    """Pipeline for TPTT model inference."""

    def __init__(self, model, tokenizer, device=None, **kwargs):
        """
        Initialize TpttPipeline.

        Args:
            model (PreTrainedModel): The TPTT model (should be a causal LM).
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            device (int or str, optional): Device to run the model on.
            **kwargs: Additional kwargs for Pipeline.
        """
        super().__init__(model=model, tokenizer=tokenizer, device=device, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        # No special parameter handling for now
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, prompt):
        # Tokenize the input prompt
        return self.tokenizer(prompt, return_tensors="pt")

    def _forward(self, model_inputs, **forward_params):
        # Move tensors to the correct device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        # Use generate for text generation
        with torch.no_grad():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=forward_params.get("max_new_tokens", 50),
                do_sample=forward_params.get("do_sample", False),
            )
        return {"generated_ids": output}

    def postprocess(self, model_outputs):
        # Decode the generated ids into text
        generated_ids = model_outputs["generated_ids"]
        return [
            {"generated_text": self.tokenizer.decode(ids, skip_special_tokens=True)}
            for ids in generated_ids
        ]
