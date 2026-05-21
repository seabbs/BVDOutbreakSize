# Estimating the current size of the 2026 DRC Bundibugyo virus outbreak: a joint Bayesian re-analysis of the McCabe et al. report

**Authors:** Sam Abbott, Samuel Brand and Sebastian Funk.

**Last updated:** 20 May 2026. This is a live report, re-run as new
data arrive, so the estimates change between updates.

[![DOI](https://zenodo.org/badge/1243778099.svg)](https://doi.org/10.5281/zenodo.20312758)

<!-- ABSTRACT:START -->
**Abstract.** An outbreak of Ebola disease caused by Bundibugyo virus
(BVD) is ongoing in the Democratic Republic of the Congo (DRC),
with cases also detected across the border in Uganda. Estimating
the likely current size of the outbreak is useful for the response,
but most
cases are not yet reported and have to be inferred from the data
streams that are available. The Imperial College London report
(McCabe et al., [18 May 2026](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/research-themes/preparedness-and-response-to-emerging-threats/report-ebola-18-05-2026/))
estimates the size with two analyses, geographic spread from the cases
exported to Uganda and back-calculation from suspected deaths in DRC.
Building on that work, we re-analyse the same problem as a single joint
Bayesian model over the latent cumulative case count, fitting
all streams together with priors on the nuisance parameters that the
report varies in scenario sweeps. Beyond the exported cases and DRC
deaths the report uses, we condition on two further streams, the
reported cases in DRC (with an ascertainment component) and the deaths
among exported cases in Uganda. We also add a no-onward-transmission
projected-deaths counterfactual, a one-week-ahead forecast and an
onset-to-death delay sensitivity analysis, and replace two closed-form
approximations (the deaths convolution and the small-growth-rate
exports term) with their exact forms. We report the joint posterior
over the cumulative case count from current data; to separate the
effect of newer data from the change in method we also fit the model
to the data as of the report, comparing against both a joint
reimplementation of the
report's approach and its original published estimates.
<!-- ABSTRACT:END -->

**Use of AI:** The model code and analysis were drafted by a language
model and reviewed and revised under human oversight; the named authors
are responsible for that oversight.

**Why our numbers differ from the Imperial report.** Two reasons.
First, the method: we fit all data streams jointly in a single
Bayesian model rather than combining separate scenario analyses (the
abstract above and the analysis page list the full set of changes).
Second, the data: we use the 20 May 2026 snapshot (sources per
[`data/observations.toml`](https://github.com/epiforecasts/BVDOutbreakSize/blob/main/data/observations.toml)),
more recent than the 16 May 2026 figures in the McCabe et al. report.
To separate these two effects we also refit the model to the report's
own data. The joint posterior assumes a single common cut-off for
every data stream, so the counts must be kept in sync to the same
date.

The [analysis](https://epiforecasts.io/BVDOutbreakSize/dev/analysis)
lays out each deviation alongside the matching Imperial method.

## Installing the package

To use the model and the bundled outbreak data from your own Julia
environment, add the package from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/epiforecasts/BVDOutbreakSize")
```

You can then load the model machinery and the data the report is
fitted to:

```julia
using BVDOutbreakSize
obs = load_observations()
```

This gives you the exported model components, constants and data
loaders, enough to build your own analysis on top of the package.
Reproducing the full report (fitting the models and writing the
result tables and plots) can be done in a few ways, described next.

## Running

There are a couple of ways to re-fit the model.

### Re-fit from a clone

```bash
git clone --recurse-submodules https://github.com/epiforecasts/BVDOutbreakSize
cd BVDOutbreakSize
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/run.jl
```

`scripts/run.jl` fits the models and writes the output CSVs (the
analysis literate is also run as part of the docs build). Running
`docs/examples/analysis.jl` directly instead steps through the full
narrative.

### Re-fit without cloning, in one command

`scripts/reproduce.jl` fetches the package, instantiates its
environment, and runs the fit, all without a manual `git clone`. Run
it straight from the web:

```bash
curl -fsSL https://raw.githubusercontent.com/epiforecasts/BVDOutbreakSize/main/scripts/reproduce.jl | julia
```

Outputs land in `./bvd-output`; set `BVD_OUTPUT_DIR` to write them
elsewhere, or `BVD_REF` to a release tag to reproduce a specific
version. The script clones into a temporary directory and runs from
there, so it leaves your own Julia environments untouched. This
indirection is needed because `Pkg.add` installs the package
read-only, while the analysis writes its outputs into the package
directory and imports several of its dependencies directly. Reading a
script before piping it to an interpreter is good practice.

### Render the docs page

Executes the literate and produces HTML at `docs/build/`:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Updating the observation counts for a new sitrep is a single-file
edit of `data/observations.toml`; the literate picks the new numbers
up automatically.

## Results

Each push to `main` regenerates the model outputs as part of the
documentation build and publishes them as a GitHub Release. The
[latest release](https://github.com/epiforecasts/BVDOutbreakSize/releases/latest)
bundles the saved result tables and plots, thinned posterior draws, a
copy of the input `observations.toml` that produced them, a
`site.zip` snapshot of the rendered report site, and `analysis.html`,
a self-contained single-file copy of the report that opens offline
([download the latest](https://github.com/epiforecasts/BVDOutbreakSize/releases/latest/download/analysis.html));
the same artifacts are
written to the repository's `output/` directory on each build. Browse
[all releases](https://github.com/epiforecasts/BVDOutbreakSize/releases)
for earlier output bundles — major versions of the report are kept as
GitHub Releases.

The rendered report is published from the
[`gh-pages` branch](https://github.com/epiforecasts/BVDOutbreakSize/tree/gh-pages),
where past and development versions of the analysis page can be found.

Regenerating these outputs locally is the "Re-fit" flow above;
`scripts/run.jl` writes them to an `output/` directory at the
repository root (`output/posterior_summary.csv`,
`cumulative_cases_by_stream.csv`, `imperial_comparison.csv`,
`scenario_coverage.csv`, `posterior_draws.csv`, and a copy of
`observations.toml`).

## Submodules

- `external/bdbv-linelist-analysis` — Bayesian reanalysis of the 2012
  Isiro BDBV line list (Rosello et al. 2015). Source of the
  informative onset-to-death gamma shape and scale priors.

## Citation

If you use or build on this project, please cite the three works
this repository depends on:

- **This project** — Abbott, S., Brand, S., Funk, S. (2026).
  *BVDOutbreakSize: joint forward-generative Turing model for the
  2026 DRC Bundibugyo outbreak.*
  <https://github.com/epiforecasts/BVDOutbreakSize>.
  DOI: [10.5281/zenodo.20312758](https://doi.org/10.5281/zenodo.20312758).
- **Imperial report** that this work re-implements —
  McCabe, R., Ebbarnezh, L., Okware, S., Fotsing, R., Koua, E.,
  Mbaka, P., Lofungola, A., van Elsland, S. L., McMenamin, M.,
  Ferguson, N., le Polain de Waroux, O., Cori, A. (2026).
  *Estimation of the size of the outbreak of Ebola disease caused
  by Bundibugyo virus in DRC.* Imperial College London, 18 May 2026.
  [Report page](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/research-themes/preparedness-and-response-to-emerging-threats/report-ebola-18-05-2026/).
- **Onset-to-death delay reanalysis** that this work uses for
  delay priors — Funk, S. (2026). *bdbv-linelist-analysis:
  Bayesian reanalysis of the 2012 Isiro Bundibugyo line list.*
  <https://github.com/sbfnk/bdbv-linelist-analysis>.

## Further references

- Rosello et al., *Ebola virus disease in DRC, 1976–2014*, eLife 2015.
  Original Isiro 2012 onset-to-death gamma point estimate.
- Imai et al., *Estimating the potential total number of novel
  coronavirus cases in Wuhan City*, Imperial COVID-19 Response Team
  Report 1, 17 January 2020. Methodological template for Method 1.
- Charniga et al., *Best practices for estimating and reporting
  epidemiological delay distributions*, PLOS Computational Biology 2024.
  Followed for the delay-distribution reporting here.
