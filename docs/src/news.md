# News

Release notes for BVDOutbreakSize.
Major versions of the report are kept as
[GitHub Releases](https://github.com/epiforecasts/BVDOutbreakSize/releases);
each push to `main` also republishes the rendered analysis and the
`output/` artifacts.

## v1.2.0

### Modelling

- Improved the comparison to the McCabe et al. report by making sure that 95% credible intervals are being compared and reordering it.
- Added a custom chain rule for `SpecialFunctions.gamma_inc`. This allows us to differentiate through the analytical solution to the gamma convolution integral.

### Data

- Moved the cut-off to 23 May 2026 and switched the DRC source from
  the WHO AFRO joint sitrep to the situation reports of the Institut
  National de Santé Publique (INSP), transcribed by
  [INRB-UMIE/Ebola_DRC_2026](https://github.com/INRB-UMIE/Ebola_DRC_2026).
  The INSP series gives a per-zone, per-sitrep daily vintage trajectory
  (suspected and confirmed; this analysis uses suspected). Cumulative
  counts at 23 May: 905 suspected DRC cases, 220 suspected DRC deaths,
  across the 12 reporting health zones. The 18 May INSP vintage (516
  cases, 131 deaths) matches the WHO joint sitrep 01 total exactly.
- Updated Uganda to three travel-related imports with one death,
  reflecting the third import announced on 23 May 2026 (woman from DRC
  who travelled Arua to Entebbe to Kampala; tested positive on
  follow-up). Two further Uganda-confirmed cases announced the same
  day (a driver and a healthcare worker) are domestic contacts of the
  first import and are excluded from `exported_cases` because the
  model treats Uganda as imports only.
- Added `reported_case_history`, `confirmed_case_history` and
  `death_history` blocks in `data/observations.toml`, six INSP sitrep
  vintages each (18 May to 23 May 2026), now consumed by the
  per-vintage likelihoods. Cumulative suspected deaths run 131 to 220
  over the window. The 14 and 15 May vintages are omitted because they
  cover only 1 and 3 reporting zones respectively.
- Added the laboratory observations from SitRep 009 (section IV.3
  LABORATOIRE): 211 cumulative tests analysed and 101 cumulative
  confirmed (PCR-positive) cases, as `cumulative_tests_analysed` and
  `confirmed_cases` in `data/observations.toml`.

### Modelling

- Added a laboratory pipeline coupling two new observations to the
  latent incidence: cumulative tests analysed (negative binomial) and
  confirmed cases (binomial on the tested pool, with per-test
  positivity the BVD share of the pool scaled by PCR sensitivity).
  Introduces a testing fraction, PCR sensitivity and a
  report-to-confirmation (lab-turnaround) delay; right-truncation of
  the tested observation is handled by the lab-delay CDF.
- Rewrote the suspected-cases stream as a BVD-driven onset-to-report
  convolution plus an additive non-BVD background rate, exposing the
  implied per-suspected positivity as a derived quantity.
- Added `confirmed_only_model`, a single-stream composer that fits the
  laboratory pipeline in isolation for the per-stream comparison.
- Fit the DRC suspected-case, laboratory-confirmed and suspected-death
  streams per sitrep vintage: `bvd_joint` now models the new cases and
  deaths reported in each sitrep, conditioning on the between-vintage
  increments rather than a single total at the cut-off, which sharpens
  the growth rate. A stream with a single vintage reduces exactly to
  the cumulative single-total likelihood, recovering the McCabe et al.
  configuration. Each case bin carries a per-bin random-effect DRC
  ascertainment, and each stream carries its own vintage offsets so a
  stream that lags or stops reporting is not assumed to run to the
  cut-off. Confirmed cases now enter as per-vintage NegBinomial
  increments with per-test positivity exposed as a derived quantity;
  the single tests-analysed count is anchored at its own date.

### Outputs

- Posterior summary table and a new laboratory-pipeline pair plot cover
  the report and lab delays, PCR sensitivity, testing fraction,
  background rate and the per-suspected and per-test positivity.
- Posterior-predictive plot gains confirmed-cases and tests-analysed
  panels, and the per-stream versus joint grid includes the laboratory
  fit.
- One-week-ahead forecast, its summary table and plots cover the
  confirmed and tests-analysed streams.
- Data table and the parameter-by-stream summary include the confirmed
  and tests-analysed streams.

### Documentation

- Surfaced the onset-to-report and report-to-lab delay priors as
  equations, and stated that the onset-to-death prior means are the
  BDBV reanalysis estimates with standard deviations reproducing its
  95% credible intervals (prior mean delay about 11 days).
- Clarified that the latent cumulative count is the true-case pool,
  not the tested or confirmed count, and framed the testing-fraction
  prior as weakly informative with no outbreak-specific data.
- Refreshed the Uganda-exports limitation for the three-import data.
- Submodel source listings render only the code (via `@eval`), no
  longer echoing the `@code_string` print statements above each block.
- Clipped the overlaid per-stream C(T) density x-axis so the
  heavy-tailed exports-deaths fit no longer compresses the other
  curves.
- Added a posterior-predictive-across-the-sitrep-series figure
  (`plot_vintage_ppc`) that reconstructs the cumulative replicate at
  each vintage and overlays the observed series, checking the fit
  against the whole trajectory rather than only the latest total.
- Distributed the per-vintage increment maths into each submodel
  section and reframed the streams as modelling the new cases and
  deaths reported in each sitrep.
- Added a limitation that the constant exponential growth rate is
  assumed to hold through and beyond the report period despite the
  response the first reports likely triggered, and that per-sitrep
  increments mix true incidence with backfill and rising ascertainment.

### Infrastructure

- Moved the submodels out of the analysis file and into the supporting package. Instead we now print these in the analysis.
- Added additional package infrastructure including `Aqua.jl` and `Jet.jl`.
- Streamlined the package unit tests.

## v1.1.0

### Modelling

- Bound the seeding time `T` from below with a soft prior on the
  genetic time to the most recent common ancestor (TMRCA), following a
  suggestion from Neil Ferguson to combine the genetic signal with the
  other data streams as a seeding bound.
- Switched the export deaths to a daily (time-resolved binned) Poisson
  process: a continuous survival weight for the no-death stretch before
  the first dated death, then a per-day Poisson from that day to the
  cut-off.
- Bound `T` with export-death timing through that survival weight, and
  with case-export detection timing through a first-export-detection
  survival term on the Uganda admission date. Dates supplied in
  `data/observations.toml`.
- Death-convolution quadrature adapted to the sampled delay scale.
- Added a clock-rate sensitivity: refit the joint model under the
  faster 1.9e-3 early-epidemic TMRCA estimate and compare the impact on
  outbreak size, seeding time and growth rate against the 1.2e-3
  baseline.
- Sped up the deaths-among-exports likelihood: precompute the
  onset-to-death CDF once and reuse it across bin edges
  (`ExportDeathDelay`), replacing the per-node nested quadrature.
- Removed hardcoded death and case constants that diverged from the
  observations in `data/observations.toml`.
- Added a forecast validation: fit the joint model to the original
  report's data, project it forward to the current cut-off, and compare
  the predicted cumulative and new counts per stream against the counts
  observed since, as a table and a 2×3 coverage plot.

### Data

- Updated to the McCabe et al. 20 May 2026 report, comparing both
  report versions.
- Sourced the genetic TMRCA seeding bound from the BEAST temporal-tree
  estimate in the 2026-05-21
  [virological.org](https://virological.org/t/initial-genomes-from-may-2026-bundibugyo-virus-disease-outbreak-in-the-democratic-republic-of-the-congo-and-uganda/1032)
  update (mean 2026-03-25, 95% HPD 2026-02-20 to 2026-04-20, at the
  1.2e-3 EBOV clock rate this analysis assumes).

### Infrastructure

- Dropped MCMCChains for FlexiChains and prepared for registry
  release.
- CI docs preview PR comments and version-bump automation.

### Docs

- Added a scope note to the README and analysis report framing the
  work as an external view built on our understanding of real-time
  infectious disease dynamics, and inviting feedback, reuse and
  adaptation.
- Surfaced results from the README and analysis landing page, added
  stable and dev docs badges.
- Plotting and labelling fixes: surveillance dispersion on the 1/√k
  scale, predictive histograms labelled as frequency, and coarser
  (four-weekly) start-date axis ticks so the labels stay readable.
- Reworked the headline summary to report the credible intervals as
  sentences rather than leading with a median, defined the prior-IQR
  shift, and explained the reported-case scaling in terms of the DRC
  reporting fraction with a link to the pair plot.
- Replaced the model-structure diagram with a parameter-to-observation
  table.
- Culled promotional register in the analysis report.

## v1.0.0

First release.
A joint Bayesian re-analysis of the McCabe et al. report that fits all
data streams together in a single Turing model over the latent
cumulative case count.

- Conditions on the exported cases and DRC deaths the report uses,
  plus reported DRC cases (with an ascertainment component) and deaths
  among exported cases.
- Adds a no-onward-transmission projected-deaths counterfactual, a
  one-week-ahead forecast of newly reported cases, deaths and exports,
  and an onset-to-death delay sensitivity analysis.
- Replaces the deaths-convolution and small-growth-rate exports
  closed-form approximations with their exact forms.
- Maths-first analysis page with code folded behind dropdowns and a
  diagram of the model build-up.
- Compares against a joint reimplementation of the report's approach
  and its original published estimates.
