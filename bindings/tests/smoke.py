from __future__ import annotations

import os
import shutil
import subprocess
import sys
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bindings" / "python" / "runtime" / "src"))

import zgc


def run(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke.py <native-zgc-executable>")
    zig = shutil.which("zig")
    if zig is None:
        raise SystemExit("zig is required for the bindings smoke test")
    native_zgc = Path(sys.argv[1]).resolve()

    with TemporaryDirectory(prefix="zgc-bindings-smoke-") as temporary:
        work = Path(temporary)
        env = os.environ.copy()
        env["ZIG_GLOBAL_CACHE_DIR"] = str(work / "zig-global")
        library = work / ("smoke.dylib" if sys.platform == "darwin" else "smoke.so")
        run(
            [
                str(native_zgc),
                "compile",
                str(ROOT / "bindings" / "tests" / "fixtures" / "smoke.zgir"),
                "--zgc-root",
                str(ROOT),
                "--zig",
                zig,
                "--cache-dir",
                str(work / "cache"),
                "--output",
                str(library),
            ],
            env=env,
        )

        with zgc.load(library) as runtime:
            assert zgc.ABI_VERSION == 2
            assert len(runtime.sources) == 2
            values = array("f", [-1.0, 2.0, -3.0, 4.0])
            runtime.bind("input", values)
            (output,) = runtime.run(bias=[0.0, 1.0, 0.0, 3.0])
            assert output.tolist() == [0.0, 3.0, 0.0, 7.0]
            values[0] = 10.0
            (output,) = runtime.run()
            assert output.tolist() == [10.0, 3.0, 0.0, 7.0]

        executable = work / "c-smoke"
        run(
            [
                zig,
                "cc",
                "-std=c11",
                "-I",
                str(ROOT / "bindings" / "c"),
                str(ROOT / "bindings" / "c" / "tests" / "smoke.c"),
                str(library),
                "-o",
                str(executable),
            ],
            env=env,
        )
        if sys.platform == "darwin":
            env["DYLD_LIBRARY_PATH"] = str(work)
        else:
            env["LD_LIBRARY_PATH"] = str(work)
        run([str(executable)], env=env)


if __name__ == "__main__":
    main()
