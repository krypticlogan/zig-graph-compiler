from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from zgc import SourceBinding, dtype

Binding: TypeAlias = SourceBinding


class Boundary(str, Enum):
    WRAP = "wrap"
    EDGE = "edge"
    REFLECT = "reflect"
    CONSTANT = "constant"


@dataclass(frozen=True)
class Redirect:
    iteration: int

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("redirect iteration must be non-negative")


SliceBoundary: TypeAlias = Boundary | Redirect


@dataclass(frozen=True)
class SliceIteration:
    offsets: tuple[int, ...]
    boundary: SliceBoundary

    def __init__(self, offsets: Sequence[int], boundary: SliceBoundary) -> None:
        normalized = tuple(offsets)
        if any(not isinstance(offset, int) or isinstance(offset, bool) for offset in normalized):
            raise TypeError("slice iteration offsets must be integers")
        if not isinstance(boundary, (Boundary, Redirect)):
            raise TypeError("invalid slice iteration boundary")
        object.__setattr__(self, "offsets", normalized)
        object.__setattr__(self, "boundary", boundary)


f32 = dtype.f32
f16 = dtype.f16
i8 = dtype.i8
bool_ = dtype.bool

owned = SourceBinding.OWNED
bound = SourceBinding.BOUND

wrap = Boundary.WRAP
edge = Boundary.EDGE
reflect = Boundary.REFLECT
constant = Boundary.CONSTANT
