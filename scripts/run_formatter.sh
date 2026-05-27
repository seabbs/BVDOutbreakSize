#!/usr/bin/env bash
# Run JuliaFormatter (SciML style) over src/, test/, docs/, scripts/
# using the project's isolated test/formatter/ sub-environment. Invoked
# by the local pre-commit hook and re-usable from the command line.
set -euo pipefail

cd "$(dirname "$0")/.."
julia --project=test/formatter -e '
using JuliaFormatter
dirs = ["src", "test", "docs", "scripts"]
all_ok = all(d -> JuliaFormatter.format(d; overwrite = true), dirs)
exit(all_ok ? 0 : 1)'
