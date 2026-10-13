"""ZGC ABI bindings and helpers."""
from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import Protocol, cast, final

ABI_VERSION = 1


class Status(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_RUNTIME_STORAGE = 2
    INVALID_SOURCE = 3
    INVALID_OUTPUT = 4
    SIZE_MISMATCH = 5
    ALIGNMENT_MISMATCH = 6
    INCOMPATIBLE_BINDING = 7
    MISSING_BINDING = 8
    BUFFER_TOO_SMALL = 9
    OUT_OF_MEMORY = 10


class dtype(IntEnum):
    f32 = 1
    f16 = 2
    i8 = 3
    bool = 4


class SourceKind(IntEnum):
    INPUT = 1
    PARAMETER = 2
    CONSTANT = 3
    STATE = 4


class SourceBinding(IntEnum):
    OWNED = 1
    BOUND = 2
    EMBEDDED = 3


@final
class TensorDescriptor(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_uint32),
        ("rank", ctypes.c_uint32),
        ("element_count", ctypes.c_size_t),
        ("logical_byte_count", ctypes.c_size_t),
        ("offset_elements", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
    ]


@final
class SourceDescriptor(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("kind", ctypes.c_uint32),
        ("binding", ctypes.c_uint32),
        ("tensor", TensorDescriptor),
    ]


def configure(library: ctypes.CDLL) -> None:
    library.zgc_abi_version.restype = ctypes.c_uint32
    library.zgc_executable_structural_fingerprint.restype = ctypes.c_uint64
    library.zgc_runtime_size.restype = ctypes.c_size_t
    library.zgc_runtime_alignment.restype = ctypes.c_size_t
    library.zgc_runtime_mutable_bytes.restype = ctypes.c_size_t

    library.zgc_runtime_init.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.zgc_runtime_init.restype = ctypes.c_int
    library.zgc_runtime_deinit.argtypes = [ctypes.c_void_p]
    library.zgc_runtime_deinit.restype = None
    library.zgc_runtime_run.argtypes = [ctypes.c_void_p]
    library.zgc_runtime_run.restype = ctypes.c_int

    library.zgc_executable_source_count.restype = ctypes.c_size_t
    library.zgc_executable_get_source_descriptor.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(SourceDescriptor),
    ]
    library.zgc_executable_get_source_descriptor.restype = ctypes.c_int
    library.zgc_runtime_copy_source.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.zgc_runtime_copy_source.restype = ctypes.c_int
    library.zgc_runtime_bind_source.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.zgc_runtime_bind_source.restype = ctypes.c_int

    library.zgc_executable_output_count.restype = ctypes.c_size_t
    library.zgc_executable_get_output_descriptor.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(TensorDescriptor),
    ]
    library.zgc_executable_get_output_descriptor.restype = ctypes.c_int
    library.zgc_runtime_copy_output.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.zgc_runtime_copy_output.restype = ctypes.c_int


class Library(Protocol):
    def zgc_abi_version(self) -> int: ...
    def zgc_executable_structural_fingerprint(self) -> int: ...
    def zgc_runtime_size(self) -> int: ...
    def zgc_runtime_alignment(self) -> int: ...
    def zgc_runtime_mutable_bytes(self) -> int: ...
    def zgc_runtime_init(self, runtime: ctypes.c_void_p, size: int) -> int: ...
    def zgc_runtime_deinit(self, runtime: ctypes.c_void_p) -> None: ...
    def zgc_runtime_run(self, runtime: ctypes.c_void_p) -> int: ...
    def zgc_executable_source_count(self) -> int: ...
    def zgc_executable_get_source_descriptor(self, index: int, output: object) -> int: ...
    def zgc_runtime_copy_source(
        self,
        runtime: ctypes.c_void_p,
        key: int,
        data: ctypes.c_void_p,
        size: int,
    ) -> int: ...
    def zgc_runtime_bind_source(
        self,
        runtime: ctypes.c_void_p,
        key: int,
        data: ctypes.c_void_p,
        size: int,
    ) -> int: ...
    def zgc_executable_output_count(self) -> int: ...
    def zgc_executable_get_output_descriptor(self, index: int, output: object) -> int: ...
    def zgc_runtime_copy_output(
        self,
        runtime: ctypes.c_void_p,
        index: int,
        data: ctypes.c_void_p,
        size: int,
    ) -> int: ...


def load_library(path: str) -> Library:
    library = ctypes.CDLL(path)
    library.zgc_abi_version.restype = ctypes.c_uint32
    version = library.zgc_abi_version()
    if version != ABI_VERSION:
        raise RuntimeError(f"unsupported ZGC ABI version {version}")
    configure(library)
    return cast(Library, cast(object, library))


def require_ok(status: int, operation: str) -> None:
    if status == Status.OK:
        return
    try:
        name = Status(status).name.lower()
    except ValueError:
        name = f"unknown status {status}"
    raise RuntimeError(f"{operation} failed: {name}")
