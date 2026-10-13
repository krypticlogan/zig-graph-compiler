from __future__ import annotations

import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import zgc as runtime
import zgc_compiler as zgc


class GraphFrontendTests(unittest.TestCase):
    def test_runtime_types_are_shared(self) -> None:
        self.assertIs(zgc.dtype, runtime.dtype)
        self.assertIs(zgc.Binding, runtime.SourceBinding)

    def test_serializes_a_canonical_dag(self) -> None:
        graph = zgc.Graph()
        inputs = graph.input("input", zgc.f32, (128, 64), binding=zgc.bound)
        weights = graph.parameter("weight", zgc.f32, (64, 10))
        result = inputs.matmul(weights).relu()
        graph.outputs(result)

        self.assertEqual(
            graph.serialize(),
            """zgir 1

%0 = input(name="input", dtype=f32, shape=[128, 64], binding=bound)
%1 = parameter(name="weight", dtype=f32, shape=[64, 10], binding=owned)
%2 = matmul(%0, %1)
%3 = relu(%2)

outputs = [%3]
""",
        )
        self.assertEqual(graph.fingerprint, graph.freeze().fingerprint)
        self.assertEqual(
            graph.fingerprint,
            hashlib.sha256(graph.serialize().encode("utf-8")).hexdigest(),
        )

    def test_direct_and_chained_calls_have_the_same_representation(self) -> None:
        direct = zgc.Graph()
        direct_x = direct.input("x", zgc.f32, (4,))
        direct_y = direct.input("y", zgc.f32, (4,))
        direct.outputs(direct.relu(direct.add(direct_x, direct_y)))

        chained = zgc.Graph()
        chained_x = chained.input("x", zgc.f32, (4,))
        chained_y = chained.input("y", zgc.f32, (4,))
        chained.outputs(chained_x.add(chained_y).relu())

        self.assertEqual(direct.serialize(), chained.serialize())

    def test_serialized_text_is_passed_to_native_compiler(self) -> None:
        source = "zgir 1\n# comment\n%0 = scalar(dtype=f32, value=3)\noutputs = [%0]\n"
        compiler = zgc.Compiler()
        seen = []

        def compile_file(path: Path, *, name: str) -> zgc.CompiledArtifact:
            seen.append((path.read_text(encoding="utf-8"), name))
            library = Path(directory) / "model.dylib"
            library.touch()
            return zgc.CompiledArtifact(library, 42)

        with TemporaryDirectory() as directory:
            index = Path(directory) / "index.json"
            with patch.object(compiler, "_graph_cache_index", return_value=index):
                with patch.object(compiler, "_compile_file", side_effect=compile_file):
                    result = compiler.compile(source)
        self.assertEqual(seen, [(source, "executable")])
        self.assertEqual(result.structural_fingerprint, 42)

    def test_cached_graph_skips_serialization_and_compilation(self) -> None:
        graph = zgc.Graph()
        graph.outputs(graph.scalar(zgc.f32, 3))
        with TemporaryDirectory() as directory:
            compiler = zgc.Compiler(zgc.CompilerConfig(cache_dir=directory))
            library = Path(directory) / "model.dylib"
            library.touch()
            index = Path(directory) / "index.json"
            index.write_text(
                '{"library": "' + str(library) + '", "library_stamp": '
                + str(list(compiler._file_stamp(library)))
                + ', "structural_fingerprint": 42}',
                encoding="utf-8",
            )
            with patch.object(compiler, "_graph_cache_index", return_value=index):
                with patch.object(compiler, "_compiled_fingerprint", return_value=42):
                    with patch.object(graph, "serialize", side_effect=AssertionError):
                        with patch.object(compiler, "_compile_file", side_effect=AssertionError):
                            artifact = compiler.compile(graph)
        self.assertEqual(artifact.path, library.resolve())

    def test_inspectable_flag_is_forwarded_and_separates_cache_entries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            definition = root / "graph.zgir"
            definition.write_text("zgir 1\noutputs = []\n", encoding="utf-8")
            library = root / "graph.dylib"
            library.touch()
            toolchain = zgc.ZigToolchain(root / "zig", "0.17.0", False)
            paths = []
            for inspectable in (False, True):
                compiler = zgc.Compiler(
                    zgc.CompilerConfig(cache_dir=root, inspectable=inspectable)
                )
                with patch.object(compiler, "toolchain", return_value=toolchain):
                    with patch.object(compiler, "_native_cli", return_value=root / "zgc"):
                        with patch.object(compiler, "_source_root", return_value=None):
                            with patch.object(compiler, "_file_stamp", return_value=(0, 0)):
                                paths.append(compiler._graph_cache_index("fingerprint", "graph"))
                    with patch.object(compiler, "_run", return_value=str(library)) as run:
                        with patch.object(compiler, "_compiled_fingerprint", return_value=42):
                            compiler.compile(definition)
                command = run.call_args.args
                self.assertEqual("--inspectable" in command, inspectable)
            self.assertNotEqual(paths[0], paths[1])


if __name__ == "__main__":
    unittest.main()
