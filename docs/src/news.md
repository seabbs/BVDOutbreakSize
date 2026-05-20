# News

Release notes for BVDOutbreakSize.
Major versions of the report are kept as
[GitHub Releases](https://github.com/epiforecasts/BVDOutbreakSize/releases);
each push to `main` also republishes the rendered analysis and the
`output/` artifacts.

## Unreleased

- Maths-first rewrite of the analysis page: explanatory text refers to
  mathematical objects throughout, with code folded behind dropdowns.
- Shared Gauss-Legendre quadrature helpers (`integrate`,
  `expected_deaths`, `integrate_cumulative`, `integrate_exports_deaths`)
  in the package, replacing the inline integrands.
- Composers conditionally include only their streams' likelihoods; new
  Imperial-exact composer (`imperial_only_model`) for the Method 2
  sense check.
- Results reordered: joint headline, counterfactual, one-week-ahead
  forecast, delay sensitivity, stream comparison, Imperial comparison,
  Imperial sense check.
- Onset-to-death delay sensitivity analysis: joint refit with the
  community-only delay alongside the baseline.
- Component-connection diagram of the model build-up.
- One-week-ahead forecast of newly reported cases, deaths and exports.
- Pooled exports / cases ascertainment and a deaths-among-exports
  likelihood.
- Model outputs published as a GitHub Release on each `main` build.
