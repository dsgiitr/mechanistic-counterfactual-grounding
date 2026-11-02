"""Circuit identifiers without model-library dependencies."""

from typing import NamedTuple


class Node(NamedTuple):
    layer_idx: int
    sequence_idx: int
    attn_head_idx: int
