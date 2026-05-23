# Proposal: using situation-report reported-case data across the redesign

Status: assessment / scoping (issue #52).
This is a data assessment, not an implementation.
It asks whether we can extract more from the situation-report (sitrep) reported-case counts than the single cumulative total we condition on now, how to do so defensibly when the timings are not trustworthy, and how well each candidate redesign architecture can use the result.

## What we condition on now

The model uses a single cumulative reported-case total: 516 DRC suspected cases as of 18-20 May 2026 (`data/observations.toml`).
The `cases_model` submodel ties it to the latent state through one scalar relation,

```
μ_c = p_DRC · C(T),     Y_cases ~ NegBinomial(μ_c, k),
```

so it reads only the endpoint `C_T = C(T)` and the DRC ascertainment fraction `p_DRC` (`pooled_ascertainment_model`), with a single shared surveillance dispersion `k` (`surveillance_dispersion_model`).
All timing in the reported-case stream is discarded.
The reported-case total therefore informs the product `p_DRC · C(T)` and nothing else; because `p_DRC` is essentially prior-driven (one aggregate point, partially pooled with the Uganda fraction), the case stream mainly pins a scale, not a shape.

## What data actually exists

Across vintages we already hold a coarse cumulative trajectory rather than a single point:

| Vintage (cut-off) | DRC suspected cases | DRC suspected deaths | File |
|---|---|---|---|
| McCabe report (16 May) | 336 | 88 | `data/report-snapshot.toml` |
| WHO AFRO sitrep 01 (18 May) | 516 | 131 | `data/observations.toml` |
| McCabe update (20 May) | 516 | 131 | `data/report-snapshot-20may.toml` |

So we have two distinct cumulative case observations (336 at 16 May, 516 at 18-20 May) and two distinct cumulative death observations (88, 131) over roughly four days, plus dated export and export-death events that the model already uses as timing bounds on `T`.

What does not yet exist in the repository, but plausibly exists or is obtainable:

- The WHO AFRO weekly external sitreps are numbered (we hold "01").
Earlier internal/WHO/MoH DRC sitreps and subsequent weekly editions would give more cumulative vintages and possibly a by-epi-week breakdown.
These are the realistic source of a genuine reported-case time series.
- A line list or epi-curve (onset or notification dates by epi-week) would be the high-value object, but the README and the analysis "Limitations" section both state none is available to us.
We should not assume one.
- Confirmed-versus-suspected splits change across vintages and would need tracking if obtained.

Recommendation on chasing data: the cheapest high-value step is to collect every dated WHO AFRO weekly sitrep and any earlier WHO DON / MoH bulletin and record each as a `(cut-off date, cumulative cases, cumulative deaths, suspected/confirmed flag)` row.
That turns the three rows above into a multi-point cumulative vintage trajectory at essentially zero modelling risk.
A by-epi-week reported-case series, if a sitrep publishes one, is the next prize but should be treated as right-truncated and backfilled (see below).
We should not fabricate weekly numbers; the trajectory should contain only what a sitrep actually prints.

## 1. What the reported-case signal can buy us, and what backfill costs

Beyond the single total, a reported-case trajectory carries three things.

Growth-rate information.
Two cumulative points (336 -> 516) over a known interval bound the recent empirical doubling time of *reported* cases.
Under the model's clean exponential-growth assumption, reported cases grow at the same rate `r` as true incidence (ascertainment cancels in a ratio if it is constant), so the vintage trajectory directly informs `r` (equivalently `τ`), which is currently almost wholly prior-driven (`τ ~ LogNormal(log 14, 0.4)`).
This is the main prize: `r` and `T = m·τ` are the ridge the whole outbreak-size estimate `C(T) = 2^m` sits on.

Elapsed-time / `T` identifiability.
Pinning `r` more tightly, combined with the existing genetic-TMRCA and export-timing bounds on `T`, narrows `C(T)`.
The case trajectory does not identify `T` on its own (it sees only recent growth, not the seed), but it sharpens the `r`-leg of the `r`-`T` trade-off.

Ascertainment-trend separation (partial).
A constant ascertainment fraction cancels in the growth ratio, so a clean trajectory informs `r` without needing `p_DRC`.
The flip side is that the trajectory cannot separate a rising ascertainment fraction from genuine epidemic growth: both inflate reported cases over time.
With only two-to-three vintages this is unidentifiable and must be handled by assumption (assume constant ascertainment over the short window, or put an explicit weak prior on an ascertainment trend and accept it is prior-driven).

What backfill and right-truncation cost.
The owner's concern is correct and decisive for the design.
The most recent cut-off in any vintage is the least complete: cases with onset before the cut-off have not yet been reported (right truncation), and earlier vintages get revised upward as backfill arrives (which is exactly the 336 -> 516 jump we see).
Consequences:

- Recent cumulative counts are downward-biased; treating them as complete biases `C(T)` and `r` down.
- A naive daily/weekly incidence series reconstructed by differencing vintages is dominated by reporting dynamics, not epidemic dynamics, near the cut-off.
- This is why we should not condition on reconstructed daily incidence or on the timing of individual reported cases.
The defensible signal is the cumulative count *at each known report date*, with the most recent vintage explicitly modelled as not-yet-complete.

## 2. How to use it defensibly: the minimal design

The minimal defensible step is to condition on the cumulative reported-case count at each known report date (the vintage trajectory), not on daily incidence and not on reported-case timings.
Concretely, generalise `cases_model` from one scalar to a small set of dated cumulative observations.
Let `d_ref` be the reference date (the latest vintage's cut-off, the date `T` is measured to) and, for each vintage `v`, let `d_v` be its cut-off date and `y_v` its cumulative count:

```
for each vintage v with cut-off date d_v and count y_v:
    s_v   = T − (d_ref − d_v)            # outbreak age at d_v (T is age at d_ref)
    μ_v   = ρ(d_v) · p_DRC · C(s_v)       # ascertained cumulative at d_v
    y_v   ~ NegBinomial(μ_v, k)
```

Three design choices make this defensible against the timing problem.

Cumulative, at known dates only.
We trust *that* a sitrep reported `y_v` cumulative cases as of `d_v`; we do not trust the implied onset timing of any individual case.
Conditioning on cumulative-at-date sidesteps the untrustworthy per-case timing entirely while still extracting the growth signal.

A completeness / right-truncation factor `ρ(d_v) ≤ 1` on the most recent vintage(s).
The earliest vintage (336 at 16 May) is nearly complete; the latest is not.
The minimal version sets `ρ = 1` for older vintages and gives the most recent vintage a completeness factor with a weakly-informative prior below 1 (or, equivalently, a known reporting-delay CDF evaluated at the elapsed time since cut-off).
This is a one-parameter nowcasting layer: it says "the latest total is a right-truncated view of `C(s_v)`", which is exactly the epinowcast framing.
If we obtain a by-epi-week series, this generalises to a reporting-delay distribution convolved against the weekly increments; with only cumulative vintages a single completeness scalar is enough and is all the data can identify.

A revision-robust observation model.
Keep the shared NegBinomial dispersion `k`, which already absorbs passive-surveillance noise, and let it also soak up vintage-to-vintage revision noise.
Do not over-fit: with two-to-three vintages, `k`, `ρ` and any ascertainment trend are weakly identified and should be read as prior-driven.

What to avoid.
Do not difference vintages into a daily/weekly incidence likelihood (amplifies backfill noise).
Do not condition on reconstructed onset timings.
Do not add a full reporting-delay distribution unless a genuine by-epi-week series arrives; the cumulative-vintage trajectory cannot identify it.

Expected payoff versus cost.
The payoff is a data-informed `r` (today prior-only) and a modestly tighter `C(T)`.
The cost is small: a vector-valued `cases_model`, one completeness parameter, and a `data/observations.toml` schema that holds a list of dated cumulative counts rather than one scalar.
This is worth doing.
A full daily-incidence or line-list nowcasting layer is not worth doing on the data we have and risks importing reporting artefacts as if they were epidemic signal.

## 3. Risks

- Backfill / right-truncation, as above: the dominant risk; mitigated by the completeness factor and by trusting cumulative-at-date rather than timing.
- Weekday and batch effects: cumulative counts jump when a batch of suspected cases is entered, so the trajectory is step-like, not smooth.
Cumulative-at-date conditioning with an overdispersed likelihood tolerates this; differenced incidence would not.
- Definition changes (suspected vs confirmed): the counts are *suspected* and the suspected/confirmed mix can shift between vintages, which moves the count for non-epidemic reasons.
Track the flag per vintage; if the definition changes mid-series, treat the affected vintages as a separate stream or widen the dispersion.
- Double-counting across vintages: the vintages are nested cumulative snapshots of the same case pool, not independent samples.
Conditioning on cumulative-at-date with a shared latent `C(s)` handles this correctly (each vintage is a thinned view of the same trajectory), but treating differenced increments as independent observations would double-count.
- Identifiability against the deaths and exports streams: deaths already inform `r` and `T` through the onset-to-death convolution, and exports through the detection window.
A reported-case trajectory that implies a different `r` than deaths/exports would surface a data conflict the current model cannot see (the "data conflict not explored" limitation).
That is a feature, but it means the case trajectory should be added with the conflict check in mind, not assumed concordant.

## 4. Fit across the candidate architectures

The decisive question is whether each architecture treats a reported-case *trajectory* as a first-class input or only its endpoint.
All four share the `growth_state` interface `(; τ, r, m, T, C_T, cumulative)`, where `cumulative` is a callable `C(s)`.
The minimal design needs `C(s_v)` at several `s_v < T`, i.e. the trajectory, not just `C_T`.
That single requirement orders the architectures.

Discrete renewal + Rt (`arch-renewal`).
Best fit.
A renewal formulation propagates incidence over a grid with a generation-interval kernel and naturally carries a time-resolved incidence/cumulative curve and a (possibly time-varying) `Rt`.
A reported-case trajectory maps directly onto cumulative-at-date observations of that curve, and the growth signal in the trajectory is exactly what informs `Rt`.
Right-truncation/backfill is the native idiom of renewal-based nowcasting (epinowcast), so the completeness factor or reporting-delay convolution drops in at the observation layer with no contortion.
This architecture makes the temporal reported-case data first-class and would also expose time-varying transmission, which the cumulative total cannot.

Continuous explicit-convolution v2 (`arch-conv-v2`, issue #5).
Strong fit.
v2 already builds every observation through explicit delay convolutions of a latent incidence curve `i_onset(t)`, including an onset-to-report convolution `E[reports] = ∫₀^T i_onset(s) F_onset_to_report(T−s) ds`.
That report-delay CDF is precisely a reporting-completeness/backfill model: evaluated at the most recent vintage it gives the right-truncation factor `ρ` for free, and the same convolution evaluated at each `d_v` gives the cumulative-at-date expectations the minimal design needs.
So v2 makes a reported-case trajectory first-class with no new machinery beyond looping the existing convolution over vintages.
The cost is the extra report-delay prior (weak data support, flagged in #5) and a second nested convolution per draw.

Stochastic latent process (`arch-stochastic`, issue #48).
Good fit, with a caveat.
The proposal (`arch-stochastic` `docs/proposals/stochastic-latent.md`) exposes the same `cumulative` trajectory as a random `C(s)`, so cumulative-at-date observations map in unchanged, and the early-phase variance it adds is genuinely useful here: it lets an early sitrep vintage sit above or below the smooth mean without forcing `r` or `T`, which is the right way to absorb backfill/batch noise in the early counts.
Caveat: the trajectory is a noisy random walk, so the process noise `σ` and the reporting/revision noise are only weakly separable from a few cumulative vintages, and `σ` would compete with the completeness factor `ρ` and dispersion `k`.
The trajectory data is usable and first-class, but the extra reporting parameters it could in principle identify will stay prior-driven.

Compartmental SEIR-style (`arch-compartmental`).
Workable but the least natural fit for *this* signal.
An SEIR model produces a time-resolved prevalence/incidence trajectory, so cumulative reported cases can be tied to cumulative incidence at each vintage in principle.
But the reported-case observation in a compartmental model is usually a thinned, delayed view of the I (or removed) compartment, and bolting a reporting-delay/backfill layer onto compartment outputs is less idiomatic than in a renewal or explicit-convolution model; it tends to require an auxiliary observation process anyway.
With early counts O(10-100) the deterministic ODE also mis-states early-phase variance (the same critique as deterministic exponential growth in #48), so the backfilled early vintages would be fit too tightly.
A stochastic/CTMC compartmental variant would fix the variance but reintroduces the discrete-latent inference problem the stochastic proposal already discusses.
Usable, but it does not make backfilled temporal reported-case data first-class without an added observation layer that the renewal and convolution tracks get for nearly free.

Verdict ordering: renewal ≈ convolution-v2 > stochastic > compartmental.
The two architectures whose observation layer is already a delay convolution of a latent incidence curve (renewal, conv-v2) absorb a right-truncated, backfilled reported-case trajectory as a first-class input; the stochastic process handles it well and adds useful early-phase slack but cannot separate the extra reporting parameters; the compartmental model can be made to fit but needs an extra observation/backfill layer and mis-handles early-phase variance.

## Recommendation

Yes, bring more sitrep reported-case information in, but only as a cumulative-at-date vintage trajectory with an explicit right-truncation/completeness factor on the most recent vintage; do not use reconstructed daily incidence or per-case timings.
The minimal defensible change to the current model is to make `cases_model` accept a list of dated cumulative counts and condition each on `C(s_v)` with a shared `k` and one completeness parameter.
The main payoff is a data-informed growth rate `r`, today essentially prior-only.
The change is most natural under the renewal and explicit-convolution-v2 architectures, where the reporting-delay/backfill layer is already present; it is workable under the stochastic process and least natural under a deterministic compartmental model.
Worth chasing as data: every dated WHO AFRO weekly sitrep and earlier WHO DON / MoH bulletin (to extend the cumulative vintage trajectory), and, if any sitrep publishes one, a by-epi-week reported-case series (treated as right-truncated and backfilled, not as clean incidence).

## Relation to issue #94

Issue #94 notes the DRC death stream is conditioned on *suspected* deaths with no death-ascertainment parameter, while the reported cases are *suspected* cases with a `p_DRC` ascertainment fraction.
The same suspected-versus-confirmed concern applies to the case trajectory proposed here: if a vintage's suspected/confirmed mix shifts, the count moves for non-epidemic reasons.
Tracking the definition flag per vintage (risk 3 above) is the case-side analogue of the death-ascertainment question in #94, and any decision there on how to treat suspected counts should be applied consistently to the case trajectory.
