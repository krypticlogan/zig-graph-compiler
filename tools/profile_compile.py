"""Probe a public Zig graph definition through one comptime checkpoint at a time."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MemoryProbe:
    def __init__(self, process: subprocess.Popen) -> None:
        self.pid = process.pid
        self.handle = None
        if os.name == "nt":
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                    )
                ]

            self.Counters = Counters
            self.open = ctypes.windll.kernel32.OpenProcess
            self.open.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            self.open.restype = wintypes.HANDLE
            self.read = ctypes.windll.psapi.GetProcessMemoryInfo
            self.read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            self.close = ctypes.windll.kernel32.CloseHandle
            self.close.argtypes = [wintypes.HANDLE]
            self.handle = self.open(0x410, False, self.pid)

    def peak_bytes(self) -> int | None:
        if self.handle:
            counters = self.Counters()
            counters.cb = ctypes.sizeof(counters)
            if self.read(self.handle, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize
        elif os.name != "nt":
            try:
                status = Path(f"/proc/{self.pid}/status").read_text()
                match = re.search(r"^VmHWM:\s+(\d+) kB", status, re.MULTILINE)
                if match:
                    return int(match[1]) * 1024
            except OSError:
                pass
        return None

    def current_bytes(self) -> int | None:
        if self.handle:
            counters = self.Counters()
            counters.cb = ctypes.sizeof(counters)
            if self.read(self.handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize
        elif os.name != "nt":
            try:
                status = Path(f"/proc/{self.pid}/status").read_text()
                match = re.search(r"^VmRSS:\s+(\d+) kB", status, re.MULTILINE)
                if match:
                    return int(match[1]) * 1024
            except OSError:
                pass
        return None

    def finish(self) -> None:
        if self.handle:
            self.close(self.handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Zig module exposing a public graph definition")
    parser.add_argument("--definition", default="definition", help="public definition declaration")
    parser.add_argument("--stage", action="append", required=True, help="stop after this Diagnostics.Stage; repeat to probe several stages")
    parser.add_argument("--detailed", action="store_true", help="log optimizer, search, and candidate substeps")
    parser.add_argument("--max-executables", type=int, default=128, help="completed search candidate allowance")
    parser.add_argument("--max-composition-attempts", type=int, default=1024, help="attempted local representation branches")
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--timeout", type=float, default=60, help="seconds per probe, including all preceding stages")
    parser.add_argument("--max-memory-mib", type=int, default=2048, help="abort the probe above this peak RSS (Windows/Linux)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".zig-cache" / "compile-diagnostics")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", args.definition):
        parser.error("--definition must be a Zig identifier")
    stage_source = (ROOT / "src/zgc/compiler/diagnostics.zig").read_text().split("pub const Stage = enum {", 1)[1].split("fn detailed", 1)[0]
    stages = set(re.findall(r"^\s*(\w+),", stage_source, re.MULTILINE))
    if any(stage not in stages for stage in args.stage):
        parser.error("unknown checkpoint; available stages: " + ", ".join(sorted(stages)))
    if args.timeout <= 0 or args.max_memory_mib <= 0:
        parser.error("timeout and memory limit must be positive")
    if args.max_executables < 1 or args.max_composition_attempts < 0:
        parser.error("search requires at least one executable and nonnegative composition attempts")
    source = args.source.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for stage in args.stage:
        harness = output / f"{stage}.zig"
        harness.write_text(
            'const model = @import("model");\ncomptime {\n'
            f'    _ = model.{args.definition}.executableWith(.{{ .search = .{{ .max_executables = {args.max_executables}, .max_composition_attempts = {args.max_composition_attempts} }}, .diagnostics = .{{ '
            f'.stop_after = .{stage}, .detailed = {str(args.detailed).lower()} }} }});\n}}\n'
        )
        command = [args.zig, "build-obj", "--dep", "model", f"-Mroot={harness}",
                   "--dep", "zgc", f"-Mmodel={source}", f"-Mzgc={ROOT / 'src/root.zig'}", "-fno-emit-bin"]
        log_path = output / f"{stage}.log"
        started = time.perf_counter()
        peak = None
        last_current = None
        samples = []
        next_sample = 0.0
        status = "compiler_error"
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
            memory = MemoryProbe(process)
            try:
                while process.poll() is None:
                    sample = memory.peak_bytes()
                    current = memory.current_bytes()
                    if current is not None:
                        last_current = current
                    elapsed_now = time.perf_counter() - started
                    if elapsed_now >= next_sample:
                        samples.append(dict(elapsed_seconds=round(elapsed_now, 3),
                                            current_rss_mib=round(current / 1024**2, 1) if current is not None else None,
                                            peak_rss_mib=round(sample / 1024**2, 1) if sample is not None else None))
                        next_sample = elapsed_now + 1
                    if sample is not None:
                        peak = max(peak or 0, sample)
                    if peak is not None and peak > args.max_memory_mib * 1024**2:
                        status = "memory_limit"
                        process.kill()
                        break
                    if time.perf_counter() - started > args.timeout:
                        status = "timeout"
                        process.kill()
                        break
                    time.sleep(0.05)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
                memory.finish()
        elapsed = time.perf_counter() - started
        diagnostics = log_path.read_text(errors="replace")
        if status == "compiler_error" and f"ZGC diagnostic stop after {stage}" in diagnostics and "Compile Log Output:" in diagnostics:
            status = "reached"
        samples_path = output / f"{stage}.memory.json"
        samples_path.write_text(json.dumps(samples, indent=2) + "\n")
        report = dict(stage=stage, status=status, elapsed_seconds=round(elapsed, 3),
                      peak_rss_mib=round(peak / 1024**2, 1) if peak is not None else None,
                      last_current_rss_mib=round(last_current / 1024**2, 1) if last_current is not None else None,
                      memory_samples=str(samples_path), returncode=process.returncode, log=str(log_path), command=command)
        reports.append(report)
        (output / "report.json").write_text(json.dumps(dict(source=str(source), probes=reports), indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if status != "reached":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
