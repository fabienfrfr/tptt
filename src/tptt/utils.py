# mypy: ignore-errors
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch


class MemoryCache:
    """
    A cache used for storing hidden states produced by models.
    It stores the states of each layer as a tensor of shape `[batch_size, key_dim, value_dim]`.
    The cache is shared between all instances of MemoryCache.
    """

    states: List[Dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        """
        Resets the shared cache for all layers.
        """
        cls.states = []

    @classmethod
    def update(
        cls,
        recurrent_state: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state` for the layer `layer_idx`.
        If the layer does not exist in the cache yet, it is created.
        """

        if len(cls.states) <= layer_idx:
            state = dict(recurrent_state=recurrent_state)
            cls.states.append(state)
        else:
            state = cls.states[layer_idx]
            if recurrent_state is not None:
                state["recurrent_state"] = recurrent_state
        return state


def extract_layer_idx(module_name: str) -> int:
    match = re.search(r"\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return -1


def instruction_format(sample):
    return {
        "text": f"### Instruction:\n{sample['instruction']}\n"
        + "\n### Input:\n{sample['input']}\n"
        + "\n### Response:\n{sample['output']}"
    }
