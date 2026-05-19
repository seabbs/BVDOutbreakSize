# BVDOutbreakSize

**This is an LLM-driven reimplementation of the Imperial / WHO BVD
report as a single joint Bayesian model. It is an experiment in
methodology and should not be used to inform public health decision
making.**

Joint forward generative Turing model for the 2026 Bundibugyo virus disease
(BVD) outbreak in the Democratic Republic of the Congo, fitting both
data sources from the Imperial / WHO report (McCabe et al.,
[18 May 2026](https://doi.org/10.25560/130007)) in a single posterior.

The original report combines two independent methods (cases exported to
Uganda; backcalculation from deaths) by running each as a sensitivity
sweep over fixed nuisance parameters.
Here we replace the sweep with priors and the closed-form approximation
with the full convolution integral, fitted jointly to both likelihoods.

## Model

Latent total cases `C_T` to date are seeded by a single zoonotic case
`T` days ago and grow exponentially at rate `r`, so `C_T = exp(r·T)`.

Two observation models share that latent state:

- **Exports**: cases detected in Uganda follow `NegBinomial(mean = C_T·p, k)`
  with `p = (daily travellers / population) × detection window`.
- **Deaths**: cumulative deaths follow `Poisson(CFR · ∫₀^T exp(r·s)·f(T−s) ds)`,
  with `f` the gamma symptom-onset-to-death density.
  The convolution is evaluated by Gauss–Legendre quadrature.

Priors are placed on the growth rate `r`, outbreak age `T`, gamma delay
shape and scale, CFR, NegBinomial dispersion, and detection window.

## Running

```bash
git clone --recurse-submodules https://github.com/seabbs/BVDOutbreakSize
cd BVDOutbreakSize
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. docs/examples/analysis.jl
```

Or render the docs page:

```bash
julia --project=docs docs/make.jl
```

## Submodules

- `external/bdbv-linelist-analysis` — Bayesian reanalysis of the 2012
  Isiro BDBV line list (Rosello et al. 2015). Source of the
  informative onset-to-death gamma prior.

## References

- McCabe et al., *Estimation of the size of the outbreak of Ebola
  disease caused by Bundibugyo virus in the Democratic Republic of the
  Congo*, Imperial College London, 18 May 2026.
  DOI: [10.25560/130007](https://doi.org/10.25560/130007).
- Rosello et al., *Ebola virus disease in the Democratic Republic of
  the Congo, 1976–2014*, eLife 2015. Used for the symptom-onset-to-death
  delay prior mean.
