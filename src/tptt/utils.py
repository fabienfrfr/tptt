"""Utility classes and functions for TPTT models (cache management, formatting, etc.)."""

# mypy: ignore-errors
from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch


class LCache:
    """
    Cache for storing intermediate states of linear attention layers.
    Supports a sliding window if max_length is set.
    """

    def __init__(self):
        """
        Initialize the cache.

        Args:
            max_length (Optional[int]): Maximum number of tokens to keep per layer (if set).
        """
        self.states: List[Dict[str, torch.Tensor]] = []
        self.seen_tokens = 0

    def __getitem__(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve the state for the given layer index, if it exists.
        """
        if layer_idx < len(self.states):
            return self.states[layer_idx]
        return None

    def update(self, layer_idx: int, **kwargs):
        """
        Update the cache for a given layer.
        If max_length is set, keep only the last max_length tokens in any sequence state.
        """
        detached_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            detached_kwargs[key] = value

        if len(self.states) <= layer_idx:
            self.states.append(detached_kwargs)
        else:
            self.states[layer_idx].update(detached_kwargs)

    def reset(self):
        """
        Reset the cache and token counter.
        """
        self.states.clear()
        self.seen_tokens = 0


def extract_layer_idx(module_name: str) -> int:
    """
    Extract the layer index from a module name string.
    """
    match = re.search(r"\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return -1
