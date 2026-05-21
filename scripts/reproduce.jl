#!/usr/bin/env julia
#
# Standalone reproduction runner. Fetches the package and re-fits the
# models without a manual `git clone`. Run it straight from the web:
#
#   curl -fsSL https://raw.githubusercontent.com/epiforecasts/BVDOutbreakSize/main/scripts/reproduce.jl | julia
#
# Outputs are written to `./bvd-output` by default; override with the
# `BVD_OUTPUT_DIR` environment variable. The analysis imports several
# of the package's dependencies directly, so it runs against the
# package's own project (obtained via `Pkg.develop`, which gives a
# writable checkout) rather than a bare `Pkg.add` environment.

using Pkg

const REPO_URL = "https://github.com/epiforecasts/BVDOutbreakSize"

Pkg.develop(url = REPO_URL)
pkg_dir = joinpath(Pkg.devdir(), "BVDOutbreakSize")

Pkg.activate(pkg_dir)
Pkg.instantiate()

output_dir = get(ENV, "BVD_OUTPUT_DIR", joinpath(pwd(), "bvd-output"))

withenv("BVD_OUTPUT_DIR" => output_dir) do
    include(joinpath(pkg_dir, "scripts", "run.jl"))
end

@info "Reproduction complete" output_dir
