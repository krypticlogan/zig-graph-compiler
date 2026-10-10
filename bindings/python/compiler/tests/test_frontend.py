from __future__ import annotations

import unittest

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

    def test_parser_round_trips(self) -> None:
        graph = zgc.Graph()
        inputs = graph.input("input", zgc.f32, (2, 2), binding=zgc.bound)
        fill = graph.scalar(zgc.f32, 0)
        shifted = inputs.shift((0, 1), boundary=zgc.constant, fill=fill)
        graph.outputs(shifted.sum(axes=(1,)))

        parsed = zgc.parse_graph(graph.serialize())
        self.assertEqual(parsed.serialize(), graph.serialize())



if __name__ == "__main__":
    unittest.main()
