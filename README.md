# BVDOutbreakSize

Joint forward generative Turing model for the 2026 Bundibugyo virus
disease outbreak in DRC, fitting the data streams from the Imperial /
WHO report (McCabe et al., [18 May 2026](https://doi.org/10.25560/130007))
in a single posterior.

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
[analysis walkthrough](docs/examples/analysis.jl) lays out each
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

## Submodules

- `external/bdbv-linelist-analysis` — Bayesian reanalysis of the 2012
  Isiro BDBV line list (Rosello et al. 2015). Source of the
  informative onset-to-death gamma shape and scale priors.

## References

- McCabe et al., *Estimation of the size of the outbreak of Ebola
  disease caused by Bundibugyo virus in DRC*, Imperial College
  London, 18 May 2026. DOI: [10.25560/130007](https://doi.org/10.25560/130007).
- Rosello et al., *Ebola virus disease in DRC, 1976–2014*, eLife
  2015. Source of the original onset-to-death gamma point estimate.
- Imai et al., *Estimating the potential total number of novel
  coronavirus cases in Wuhan City*, Imperial COVID-19 Response Team
  Report 1, 17 January 2020. Methodological template for Method 1.
- Charniga et al., *Best practices for estimating and reporting
  epidemiological delay distributions*, PLOS Computational Biology
  2024. Followed for the delay-distribution reporting here.
