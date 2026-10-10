from __future__ import annotations

import ctypes
import struct
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal, Protocol, Self, cast, final

from ._abi import (
    ABI_VERSION,
    Library,
    SourceBinding,
    SourceDescriptor,
    SourceKind,
    TensorDescriptor,
    dtype as dtype_enum,
    load_library,
    require_ok,
)


class _TensorDescriptorFields(Protocol):
    dtype: int
    rank: int
    element_count: int
    logical_byte_count: int
    offset_elements: int
    shape: Sequence[int]
    strides: Sequence[int]


class _SourceDescriptorFields(Protocol):
    key: int
    name: bytes | None
    kind: int
    binding: int
    tensor: TensorDescriptor


@dataclass(frozen=True)
class TensorInfo:
    dtype: dtype_enum
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    element_count: int
    byte_count: int
    offset: int


@dataclass(frozen=True)
class Source:
    key: int
    name: str
    kind: SourceKind
    binding: SourceBinding
    tensor: TensorInfo


@dataclass(frozen=True)
class Tensor:
    dtype: dtype_enum
    shape: tuple[int, ...]
    _storage: bytearray

    @property
    def buffer(self) -> memoryview:
        raw = memoryview(self._storage)
        return cast(
            memoryview,
            raw.cast(  # pyright: ignore[reportCallIssue]
                _format(self.dtype),  # pyright: ignore[reportArgumentType]
                shape=list(self.shape),
            ),
        )

    @property
    def values(self) -> tuple[float | int | bool, ...]:
        return _decode(self.dtype, len(self._storage), self._storage)

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self.buffer.strides or ())

    def tolist(self) -> list[float | int | bool]:
        return list(self.values)


def _tensor_info(descriptor: TensorDescriptor) -> TensorInfo:
    fields = cast(_TensorDescriptorFields, cast(object, descriptor))
    return TensorInfo(
        dtype=dtype_enum(fields.dtype),
        shape=tuple(fields.shape[index] for index in range(fields.rank)),
        strides=tuple(fields.strides[index] for index in range(fields.rank)),
        element_count=fields.element_count,
        byte_count=fields.logical_byte_count,
        offset=fields.offset_elements,
    )


def _format(dtype: dtype_enum) -> Literal["f", "e", "b", "?"]:
    if dtype is dtype_enum.f32:
        return "f"
    if dtype is dtype_enum.f16:
        return "e"
    if dtype is dtype_enum.i8:
        return "b"
    return "?"


def _item_size(dtype: dtype_enum) -> int:
    return struct.calcsize(f"={_format(dtype)}")


def _decode(
    dtype: dtype_enum,
    byte_count: int,
    payload: bytes | bytearray | memoryview,
) -> tuple[float | int | bool, ...]:
    element_count = byte_count // _item_size(dtype)
    return struct.unpack(f"={element_count}{_format(dtype)}", payload)


def _native_format(view: memoryview) -> str | None:
    value = view.format
    if not value:
        return None
    if value[0] in "@=":
        return value[1:]
    if value[0] in "<>!":
        if view.itemsize == 1:
            return value[1:]
        native_prefix = "<" if sys.byteorder == "little" else ">"
        return value[1:] if value[0] == native_prefix else None
    return value


def _buffer_view(info: TensorInfo, values: object) -> memoryview | None:
    try:
        view = memoryview(values)  # pyright: ignore[reportArgumentType]
    except TypeError:
        return None

    if view.nbytes != info.byte_count:
        raise ValueError(
            f"expected {info.byte_count} source bytes, received {view.nbytes}"
        )

    # Byte-oriented buffers preserve the raw-byte input supported by the ABI.
    if view.itemsize == 1 and _native_format(view) in {"b", "B", "c"}:
        return view

    expected_format = _format(info.dtype)
    if _native_format(view) != expected_format or view.itemsize != _item_size(
        info.dtype
    ):
        raise TypeError(
            f"expected a buffer with format {expected_format!r} for {info.dtype.name.lower()}"
        )

    shape = tuple(view.shape or ())
    if shape != info.shape and shape != (info.element_count,):
        raise ValueError(f"expected source shape {info.shape}, received {shape}")
    return view


def _logical_bytes(info: TensorInfo, values: object) -> bytes:
    view = _buffer_view(info, values)
    if view is not None:
        return view.tobytes(order="C")

    try:
        materialized = tuple(cast(Iterable[object], values))
    except TypeError as error:
        raise TypeError(
            "source must be a buffer or an iterable of scalar values"
        ) from error
    if len(materialized) != info.element_count:
        message = (
            f"expected {info.element_count} elements for shape {info.shape}; "
            f"received {len(materialized)}"
        )
        raise ValueError(message)
    return struct.pack(f"={info.element_count}{_format(info.dtype)}", *materialized)


def _writable_pointer(
    view: memoryview,
) -> tuple[ctypes.c_ubyte, ctypes.c_void_p] | None:
    if view.readonly or not view.c_contiguous or view.nbytes == 0:
        return None
    anchor = ctypes.c_ubyte.from_buffer(view)
    return anchor, ctypes.c_void_p(ctypes.addressof(anchor))


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)


def _physical_buffer_matches(info: TensorInfo, view: memoryview) -> bool:
    if info.offset != 0 or not view.c_contiguous:
        return False
    if view.itemsize == 1 and _native_format(view) in {"b", "B", "c"}:
        return (
            view.ndim == 1
            and view.nbytes == info.byte_count
            and info.strides == _contiguous_strides(info.shape)
        )
    expected_strides = tuple(stride * view.itemsize for stride in info.strides)
    return (
        tuple(view.shape or ()) == info.shape
        and tuple(view.strides or ()) == expected_strides
    )


def _pack_bound(info: TensorInfo, logical: bytes) -> bytes:
    logical_values = _decode(info.dtype, info.byte_count, logical)
    physical_values: list[float | int | bool] = [0] * info.element_count
    for linear_index, value in enumerate(logical_values):
        remaining = linear_index
        physical_index = info.offset
        for axis in range(len(info.shape) - 1, -1, -1):
            coordinate = remaining % info.shape[axis]
            remaining //= info.shape[axis]
            physical_index += coordinate * info.strides[axis]
        if physical_index < 0 or physical_index >= info.element_count:
            raise ValueError("bound source layout exceeds its ABI storage extent")
        physical_values[physical_index] = value
    return struct.pack(
        f"={info.element_count}{_format(info.dtype)}",
        *physical_values,
    )


def _aligned_storage(
    size: int,
    alignment: int,
) -> tuple[ctypes.Array[ctypes.c_ubyte], ctypes.c_void_p]:
    if alignment <= 0 or alignment & (alignment - 1):
        raise RuntimeError(f"invalid model alignment: {alignment}")
    allocation = (ctypes.c_ubyte * (size + alignment - 1))()
    address = (ctypes.addressof(allocation) + alignment - 1) & -alignment
    return allocation, ctypes.c_void_p(address)


@final
class Model:
    def __init__(self, path: str | PathLike[str]) -> None:
        self._library: Library = load_library(str(Path(path).expanduser().resolve()))
        version = self._library.zgc_abi_version()
        if version != ABI_VERSION:
            raise RuntimeError(f"unsupported ZGC ABI version {version}")

        self.model_size: int = self._library.zgc_model_size()
        self.model_alignment: int = self._library.zgc_model_alignment()
        self.mutable_bytes: int = self._library.zgc_model_mutable_bytes()
        self._allocation, self._model = _aligned_storage(
            self.model_size, self.model_alignment
        )
        self._closed = False
        self._bound_storage: dict[int, object] = {}
        require_ok(
            self._library.zgc_model_init(self._model, self.model_size),
            "zgc_model_init",
        )
        self.sources = self._read_sources()
        self.outputs = self._read_outputs()
        self._sources_by_name = {source.name: source for source in self.sources}

    def _read_sources(self) -> tuple[Source, ...]:
        sources: list[Source] = []
        for index in range(self._library.zgc_source_count()):
            descriptor = SourceDescriptor()
            require_ok(
                self._library.zgc_get_source_descriptor(
                    index, ctypes.byref(descriptor)
                ),
                "zgc_get_source_descriptor",
            )
            fields = cast(_SourceDescriptorFields, cast(object, descriptor))
            sources.append(
                Source(
                    key=fields.key,
                    name=fields.name.decode() if fields.name is not None else "",
                    kind=SourceKind(fields.kind),
                    binding=SourceBinding(fields.binding),
                    tensor=_tensor_info(fields.tensor),
                )
            )
        return tuple(sources)

    def _read_outputs(self) -> tuple[TensorInfo, ...]:
        outputs: list[TensorInfo] = []
        for index in range(self._library.zgc_output_count()):
            descriptor = TensorDescriptor()
            require_ok(
                self._library.zgc_get_output_descriptor(
                    index, ctypes.byref(descriptor)
                ),
                "zgc_get_output_descriptor",
            )
            outputs.append(_tensor_info(descriptor))
        return tuple(outputs)

    def set_source(
        self,
        name: str,
        values: object,
    ) -> None:
        """Copy logical source values into storage owned by the model.

        ``values`` may implement the Python buffer protocol or be an iterable
        of scalar values. For a compiled bound source, the values are packed
        into retained internal storage. Later changes to ``values`` therefore
        cannot affect the model. Use :meth:`bind` to explicitly share caller
        storage without copying.
        """
        self._require_open()
        try:
            source = self._sources_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown model source {name!r}") from error
        if source.binding is SourceBinding.EMBEDDED:
            raise ValueError(f"source {name!r} is embedded and cannot be replaced")

        if source.binding is SourceBinding.BOUND:
            payload = _pack_bound(
                source.tensor,
                _logical_bytes(source.tensor, values),
            )
            storage = bytearray(payload)
            storage_view = memoryview(storage)
            address = _writable_pointer(storage_view)
            if address is None:
                raise ValueError("cannot bind empty source storage")
            anchor, pointer = address
            require_ok(
                self._library.zgc_model_bind_source(
                    self._model,
                    source.key,
                    pointer,
                    len(storage),
                ),
                "zgc_model_bind_source",
            )
            self._bound_storage[source.key] = (storage, storage_view, anchor)
            return

        view = _buffer_view(source.tensor, values)
        direct = _writable_pointer(view) if view is not None else None
        if direct is not None and view is not None and view.c_contiguous:
            anchor, pointer = direct
            byte_count = view.nbytes
            keepalive: object = (values, view, anchor)
        else:
            payload = _logical_bytes(source.tensor, values)
            pointer = ctypes.cast(ctypes.c_char_p(payload), ctypes.c_void_p)
            byte_count = len(payload)
            keepalive = payload
        require_ok(
            self._library.zgc_model_copy_source(
                self._model,
                source.key,
                pointer,
                byte_count,
            ),
            "zgc_model_copy_source",
        )
        _ = keepalive

    def bind(self, name: str, values: object) -> None:
        """Bind caller-owned physical storage without copying.

        ``values`` must export a writable, contiguous buffer whose metadata exactly match the
        compiled bound source. The model retains the buffer until the source
        is rebound or the model is closed.
        """
        self._require_open()
        try:
            source = self._sources_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown model source {name!r}") from error
        if source.binding is not SourceBinding.BOUND:
            raise ValueError(f"source {name!r} is not a bound source")

        view = _buffer_view(source.tensor, values)
        if view is None:
            raise TypeError("bound sources require an object exporting a buffer")
        if not _physical_buffer_matches(source.tensor, view):
            raise ValueError("buffer does not match the compiled source layout")
        address = _writable_pointer(view)
        if address is None:
            raise ValueError("zero-copy binding requires a writable contiguous buffer")
        anchor, pointer = address
        require_ok(
            self._library.zgc_model_bind_source(
                self._model,
                source.key,
                pointer,
                view.nbytes,
            ),
            "zgc_model_bind_source",
        )
        self._bound_storage[source.key] = (values, view, anchor)

    def run(self, **sources: object) -> tuple[Tensor, ...]:
        self._require_open()
        for name, values in sources.items():
            self.set_source(name, values)
        require_ok(self._library.zgc_model_run(self._model), "zgc_model_run")
        return tuple(self.output(index) for index in range(len(self.outputs)))


    def __call__(
        self,
        **sources: object,
    ) -> Tensor | tuple[Tensor, ...]:
        """Call the model with it's sources to run and recieve an output"""
        outputs = self.run(**sources)
        return outputs[0] if len(outputs) == 1 else outputs

    def output(self, index: int) -> Tensor:
        """Copied output from the model."""
        self._require_open()
        if index < 0:
            raise IndexError(f"model has no output {index}")
        try:
            info = self.outputs[index]
        except IndexError as error:
            raise IndexError(f"model has no output {index}") from error
        storage = bytearray(info.byte_count)
        view = memoryview(storage)
        address = _writable_pointer(view)
        if address is None:
            raise ValueError("cannot create empty output storage")
        anchor, pointer = address
        require_ok(
            self._library.zgc_model_copy_output(
                self._model,
                index,
                pointer,
                info.byte_count,
            ),
            "zgc_model_copy_output",
        )
        _ = anchor
        return Tensor(info.dtype, info.shape, storage)

    def close(self) -> None:
        if self._closed:
            return
        self._library.zgc_model_deinit(self._model)
        self._bound_storage.clear()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("model is closed")

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()


def load(path: str | PathLike[str]) -> Model:
    """Load a pre-compiled model from a local path."""
    return Model(path)
