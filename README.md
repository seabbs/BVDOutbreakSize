# Replicating and expanding the Imperial 2026 DRC Bundibugyo outbreak analysis with joint Bayesian modelling

Joint generative Turing model for the 2026 Bundibugyo virus disease
(BVD) outbreak in the Democratic Republic of the Congo, fitting the
data streams from the Imperial / WHO report (McCabe et al.,
[18 May 2026](https://doi.org/10.25560/130007)) in a single Bayesian
posterior over the latent cumulative case count `C(T)`. The original
report runs two independent analyses — geographic spread from cases
detected in Uganda, and back-calculation from suspected deaths in DRC
— and sweeps over fixed nuisance parameters. Here those nuisance
parameters carry priors, all streams are conditioned on jointly, the
closed-form deaths approximation is replaced with the full gamma
convolution and the small-growth-rate exports simplification with the
exact cumulative integral, and a reported-case ascertainment
extension, a no-onward-transmission projected-deaths counterfactual, a
one-week-ahead forecast and an onset-to-death delay sensitivity
analysis are added.

**Authors:** Sam Abbott and contributors.
The model code and analysis were drafted by a language model and
reviewed and revised under human oversight; the named authors are
responsible for that oversight.

**Data last updated:** 20 May 2026 (sources per
[`data/observations.toml`](https://github.com/epiforecasts/BVDOutbreakSize/blob/main/data/observations.toml)).
These are different, more recent figures than the McCabe et al. report,
which uses the 16 May 2026 snapshot. The joint posterior assumes a
single common cut-off for every data stream, so the counts must be kept
in sync to the same date.

The [analysis](https://epiforecasts.github.io/BVDOutbreakSize/analysis/)
lays out each deviation alongside the matching Imperial method.

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

Each push to `main` regenerates the model outputs and publishes them
as a GitHub Release. The
[latest release](https://github.com/epiforecasts/BVDOutbreakSize/releases/latest)
bundles the saved result tables and plots, thinned posterior draws, a
copy of the input `observations.toml` that produced them, and a
`site.zip` snapshot of the rendered report site; the same artifacts are
written to the repository's `output/` directory on each build. Browse
[all releases](https://github.com/epiforecasts/BVDOutbreakSize/releases)
for earlier output bundles — major versions of the report are kept as
GitHub Releases.

The rendered report is published from the
[`gh-pages` branch](https://github.com/epiforecasts/BVDOutbreakSize/tree/gh-pages),
where past and development versions of the analysis page can be found.

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
