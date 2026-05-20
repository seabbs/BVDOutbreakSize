# Entry point for regenerating the published results.
#
# Runs the full analysis literate, which fits the models and writes
# the summary tables, thinned posterior draws, and a copy of the
# input data into `output/` at the repo root. The Release workflow
# bundles that directory into a GitHub Release on each push to
# `main`.

using BVDOutbreakSize

const REPO_ROOT = pkgdir(BVDOutbreakSize)

include(joinpath(REPO_ROOT, "docs", "examples", "analysis.jl"))
