from __future__ import annotations

# pyright: reportPrivateUsage=false
import ctypes
import struct
import unittest
from array import array
from typing import final

from zgc import Runtime, Source, SourceBinding, SourceKind, Tensor, TensorInfo, dtype


@final
class _Library:
    copied_source: bytes = b""
    bound_pointer: int = 0
    bound_size: int = 0
    output: bytes = struct.pack("=4f", 5.0, 6.0, 7.0, 8.0)

    def zgc_runtime_copy_source(
        self,
        _runtime: ctypes.c_void_p,
        _source_key: int,
        data: ctypes.c_void_p,
        size: int,
    ) -> int:
        self.copied_source = ctypes.string_at(data, size)
        return 0

    def zgc_runtime_bind_source(
        self,
        _runtime: ctypes.c_void_p,
        _source_key: int,
        data: ctypes.c_void_p,
        size: int,
    ) -> int:
        self.bound_pointer = data.value or 0
        self.bound_size = size
        return 0

    def zgc_runtime_copy_output(
        self,
        _runtime: ctypes.c_void_p,
        _output_index: int,
        destination: ctypes.c_void_p,
        size: int,
    ) -> int:
        _result = ctypes.memmove(
            destination,
            self.output,
            min(size, len(self.output)),
        )
        return 0

    def zgc_runtime_deinit(self, _: ctypes.c_void_p) -> None:
        pass


def _runtime(binding: SourceBinding) -> tuple[Runtime, _Library]:
    info = TensorInfo(dtype.f32, (2, 2), (2, 1), 4, 16, 0)
    source = Source(0, "input", SourceKind.INPUT, binding, info)
    library = _Library()
    runtime = Runtime.__new__(Runtime)
    runtime._library = library  # pyright: ignore[reportAttributeAccessIssue]
    runtime._runtime = ctypes.c_void_p(1)
    runtime._closed = False
    runtime._bound_storage = {}
    runtime._sources_by_name = {"input": source}
    runtime.outputs = (info,)
    return runtime, library


class BufferTests(unittest.TestCase):
    def test_list_source_uses_scalar_fallback(self) -> None:
        runtime, library = _runtime(SourceBinding.OWNED)
        runtime.set_source("input", [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(library.copied_source, struct.pack("=4f", 1, 2, 3, 4))

    def test_array_source_uses_buffer_contents(self) -> None:
        runtime, library = _runtime(SourceBinding.OWNED)
        values = array("f", [1.0, 2.0, 3.0, 4.0])
        runtime.set_source("input", values)
        self.assertEqual(library.copied_source, values.tobytes())

    def test_bound_array_remains_borrowed(self) -> None:
        runtime, library = _runtime(SourceBinding.BOUND)
        values = array("f", [1.0, 2.0, 3.0, 4.0])
        shaped = memoryview(values).cast("B").cast("f", shape=[2, 2])
        runtime.bind("input", shaped)
        values[0] = 9.0
        payload = ctypes.string_at(library.bound_pointer, library.bound_size)
        self.assertEqual(payload, values.tobytes())

    def test_set_source_copies_bound_values(self) -> None:
        runtime, library = _runtime(SourceBinding.BOUND)
        values = array("f", [1.0, 2.0, 3.0, 4.0])
        runtime.set_source("input", values)
        values[0] = 9.0
        payload = ctypes.string_at(library.bound_pointer, library.bound_size)
        self.assertEqual(payload, struct.pack("=4f", 1.0, 2.0, 3.0, 4.0))

    def test_output_owns_a_shaped_buffer(self) -> None:
        runtime, _ = _runtime(SourceBinding.OWNED)
        output = runtime.output(0)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.buffer.format, "f")
        self.assertEqual(output.buffer.shape, (2, 2))
        self.assertEqual(output.strides, (8, 4))
        self.assertEqual(output.tolist(), [5.0, 6.0, 7.0, 8.0])


if __name__ == "__main__":
    _ = unittest.main()
