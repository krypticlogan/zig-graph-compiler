from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]


class CompileDiagnosticsTests(unittest.TestCase):
    def test_checkpoints_stop_before_later_pipeline_work(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "model.zig"
            model.write_text('''const zgc = @import("zgc");
pub const definition = blk: {
    var builder = zgc.DefinitionBuilder.init();
    const sources = builder.sources(enum { input });
    const input = sources.input(.input, .f32, &.{4});
    builder.output(builder.relu(builder.neg(input)));
    break :blk builder.finish();
};
''')
            for stage, later in (
                ("raw_validation", "semantic_optimization"),
                ("canonical_tensor_order", "canonical_rewrite"),
                ("reference_scheduling", "reference_construction"),
                ("reference_construction", "candidate_validation"),
                ("search_lowering", "search_scheduling"),
                ("candidate_storage", "candidate_cost"),
                ("executable_generation", None),
            ):
                with self.subTest(stage=stage):
                    output = directory / stage
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "tools/profile_compile.py"), str(model),
                         "--stage", stage, "--detailed", "--output-dir", str(output)],
                        capture_output=True, text=True, timeout=75,
                    )
                    failure_log = output / f"{stage}.log"
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr + (failure_log.read_text(errors="replace") if failure_log.exists() else ""))
                    report = json.loads((output / "report.json").read_text())
                    self.assertEqual(report["probes"][0]["status"], "reached")
                    samples = json.loads(Path(report["probes"][0]["memory_samples"]).read_text())
                    self.assertTrue(samples)
                    self.assertIn("current_rss_mib", samples[0])
                    self.assertIn("peak_rss_mib", samples[0])
                    log = (output / f"{stage}.log").read_text()
                    self.assertIn(f"ZGC diagnostic stop after {stage}", log)
                    self.assertIn('"definition"', log)
                    self.assertIn("Compile Log Output:", log)
                    if later is not None:
                        self.assertNotIn(f'"{later}"', log)

    def test_invalid_checkpoint_is_rejected_before_compiling(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/profile_compile.py"), "missing.zig",
             "--stage", "misspelled"], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown checkpoint", result.stderr)


if __name__ == "__main__":
    unittest.main()
