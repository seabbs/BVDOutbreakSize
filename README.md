# Replicating and expanding the Imperial 2026 DRC Bundibugyo outbreak analysis with joint Bayesian modelling

Joint generative Turing model for the 2026 Bundibugyo virus
disease outbreak in DRC, fitting the data streams from the Imperial /
WHO report (McCabe et al., [18 May 2026](https://doi.org/10.25560/130007))
in a single posterior.

**Data last updated:** 19 May 2026 (sources per
[`data/observations.toml`](https://github.com/epiforecasts/BVDOutbreakSize/blob/main/data/observations.toml)).

The original report runs two independent analyses — geographic spread
from cases detected in Uganda, and backcalculation from deaths — and
reports a sensitivity sweep over fixed nuisance parameters. Here those
nuisance parameters carry priors, all streams are conditioned on
jointly, and the output is one posterior over cumulative cases `C_T`.

What it does differently: replaces Imperial's sensitivity sweep
over fixed nuisance parameters with priors, replaces the closed-form
deaths approximation with the full gamma convolution, replaces the
small-growth-rate exports simplification with the exact cumulative
integral, and adds a reported-case ascertainment extension and a
no-onward-transmission projected-deaths counterfactual. The
[analysis walkthrough](https://epiforecasts.github.io/BVDOutbreakSize/analysis/) lays out each
deviation alongside the matching Imperial method.

## Running

```bash
git clone --recurse-submodules https://github.com/seabbs/BVDOutbreakSize
cd BVDOutbreakSize
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. docs/examples/analysis.jl
```

Render the docs page (executes the literate and produces HTML at
`docs/build/`):

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
bundles the posterior summary tables, thinned posterior draws, and a
copy of the input `observations.toml` that produced them.

To regenerate the same outputs locally:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/run.jl
```

This fits the model and writes the CSVs to an `output/` directory at
the repository root (`output/posterior_summary.csv`,
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

- **This project** — Abbott, S. (2026). *BVDOutbreakSize: joint
  forward-generative Turing model for the 2026 DRC Bundibugyo
  outbreak.* <https://github.com/epiforecasts/BVDOutbreakSize>.
- **Imperial / WHO report** that this work re-implements —
  McCabe, R., Ebbarnezh, L., Okware, S., Fotsing, R., Koua, E.,
  Mbaka, P., Lofungola, A., van Elsland, S. L., McMenamin, M.,
  Ferguson, N., le Polain de Waroux, O., Cori, A. (2026).
  *Estimation of the size of the outbreak of Ebola disease caused
  by Bundibugyo virus in DRC.* Imperial College London, 18 May 2026.
  DOI: [10.25560/130007](https://doi.org/10.25560/130007).
- **Onset-to-death delay reanalysis** that this work uses for
  delay priors — Funk, S. (2026). *bdbv-linelist-analysis:
  Bayesian reanalysis of the 2012 Isiro Bundibugyo line list.*
  <https://github.com/sbfnk/bdbv-linelist-analysis>.

## Further references

- Rosello et al., *Ebola virus disease in DRC, 1976–2014*, eLife
  2015. Original Isiro 2012 onset-to-death gamma point estimate.
- Imai et al., *Estimating the potential total number of novel
  coronavirus cases in Wuhan City*, Imperial COVID-19 Response Team
  Report 1, 17 January 2020. Methodological template for Method 1.
- Charniga et al., *Best practices for estimating and reporting
  epidemiological delay distributions*, PLOS Computational Biology
  2024. Followed for the delay-distribution reporting here.
