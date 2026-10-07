#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(dirname -- "$script_dir")

cd "$repository_root"

for workload in contraction reduction fusion dense lbm conway; do
    printf '\n%s\n' "Running benchmark=${workload}"
    zig build benchmark \
        -Dbenchmark="$workload" \
        -Doptimize=ReleaseFast \
        "$@"
done
