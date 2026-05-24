## Compartmental architecture (branch `arch-compartmental-mtk`).
##
## Candidate redesign that replaces the deterministic exponential
## cumulative-incidence trajectory $C(s) = e^{rs}$ (and the renewal
## candidate's daily recursion) with an SEIRD compartmental latent
## process specified symbolically as a Catalyst.jl reaction network.
##
## Two parts:
##   1. The Catalyst.jl symbolic specification (`bvd_seir_network`), the
##      composable, first-class description of the compartments and
##      flows. This is the artefact the modelling owner wanted: extending
##      the latent process to add a compartment (hospitalisation, vaccine
##      stratum) is one reaction line away.
##   2. A daily semi-implicit Euler stepper (`step_seir_daily`) that
##      consumes the same rate-law information and forward-propagates the
##      state on a daily grid. The stepper is the AD-friendly path used
##      by the Turing model; routing NUTS through the continuous
##      OrdinaryDiffEq solve does not work under Mooncake (verdict
##      recorded in docs/src/proposals/compartmental-mtk.md).
##
## Names used below (`@model`, `to_submodel`, `Normal`, `LogNormal`,
## `truncated`, `Poisson`, `Beta`, `logit`, `logistic`, `Gamma`, `pdf`,
## `NegativeBinomial`, the `ITURI_*` constants, `Catalyst`,
## `ModelingToolkit`, `OrdinaryDiffEq`) are imported by the enclosing
## module, into which this file is `include`d. This file deliberately
## carries no `using`/`import` statements; the module page owns imports.

## --- Symbolic Catalyst specification ------------------------------------

"""
$(TYPEDSIGNATURES)

Catalyst.jl reaction network specifying the latent BVD compartmental
process. Five compartments (`S, E, I, R, D`) and four reactions:

```
β/N:           S + I --> E + I
σ:             E     --> I
(1 - CFR)·γ:   I     --> R
CFR·γ:         I     --> D
```

`β` is the transmission rate, `σ` the E -> I rate (`1/σ` the latent
period), `γ` the I -> {R, D} rate (`1/γ` the infectious period), `CFR`
the case-fatality ratio, and `N` the source-area population. The
return is a `complete`-d Catalyst `ReactionSystem` so it can be
converted to an `ODESystem`, queried symbolically, or extended by
reaction-network composition before the forward map is built.
"""
function bvd_seir_network()
    return Catalyst.complete(
        Catalyst.@reaction_network bvd_seir begin
            β / N, S + I --> E + I
            σ,     E     --> I
            (1 - CFR) * γ, I --> R
            CFR * γ,       I --> D
        end
    )
end

"""
$(TYPEDSIGNATURES)

Reference continuous-time ODE forward map: lower
[`bvd_seir_network`](@ref) to an `ModelingToolkit` `ODESystem`, build
an `ODEProblem` over `[0, T]` and solve. Used for validation
(prior-predictive plots, sanity check against the daily stepper) and
NOT used inside the Turing model; Mooncake cannot differentiate
through this path (see the AD-route verdict in the proposal).
"""
function bvd_seir_ode_solve(T::Real;
        β::Real, σ::Real, γ::Real, CFR::Real, N::Real,
        S0::Real, E0::Real, I0::Real, R0::Real = zero(N),
        D0::Real = zero(N), saveat::Real = one(T))
    rn = bvd_seir_network()
    u0  = [rn.S => float(S0), rn.E => float(E0),
           rn.I => float(I0), rn.R => float(R0),
           rn.D => float(D0)]
    pv  = [rn.β => float(β), rn.σ => float(σ),
           rn.γ => float(γ), rn.CFR => float(CFR),
           rn.N => float(N)]
    prob = OrdinaryDiffEq.ODEProblem(rn, [u0; pv], (zero(T), float(T)))
    return OrdinaryDiffEq.solve(prob, OrdinaryDiffEq.Tsit5();
                                saveat = saveat)
end

## --- Daily semi-implicit Euler stepper ----------------------------------
##
## $(1 - e^{-r})$ is the exact transition probability for a constant-rate
## exponential clock over a unit interval. Using it for every flow
## preserves $S + E + I + R + D = N$ exactly and degenerates smoothly to
## the linear Euler step for small rates. AD-transparent under Mooncake:
## only multiplication, subtraction and `exp`, no division by latent
## quantities and no solver internals.

# NaN/Inf-safe positive rate. Extreme NUTS warmup proposals can push
# intermediate quantities transiently non-finite; `max(x, eps)` would
# propagate a NaN through `max(NaN, eps) = NaN` and trip the downstream
# Poisson/NegBinomial domain check.
@inline _seir_safe_pos(x) =
    isfinite(x) ? max(x, eps(typeof(x))) : eps(typeof(x))

"""
$(TYPEDSIGNATURES)

Advance the SEIRD state by one day under the semi-implicit Euler rule.
`state = (S, E, I, R, D)`; kwargs are the sampled rates and the
population. Returns `(next_state, new_onsets, new_deaths)`: the daily
\$E \\to I\$ flux (onsets, the observation seam) and the daily
\$I \\to D\$ flux (deaths) are exposed by the stepper itself rather
than recomputed downstream.
"""
@inline function step_seir_daily(state; β, σ, γ, CFR, N)
    S, E, I, R, D = state
    λ      = β * I / N
    pSE    = -expm1(-λ)
    pEI    = -expm1(-σ)
    pIout  = -expm1(-γ)

    new_infections = pSE * S
    new_onsets     = pEI * E
    out_I          = pIout * I
    new_deaths     = CFR * out_I
    new_recoveries = out_I - new_deaths

    Sn = S - new_infections
    En = E + new_infections - new_onsets
    In = I + new_onsets     - out_I
    Rn = R + new_recoveries
    Dn = D + new_deaths
    return (Sn, En, In, Rn, Dn), new_onsets, new_deaths
end

"""
$(TYPEDSIGNATURES)

Run the daily stepper for `n` days from initial state
`(S0, E0, I0, R0, D0)`, returning the length-`n` vectors of daily
onsets (the \$E \\to I\$ flux) and daily new deaths
(the \$I \\to D\$ flux). The state itself is not returned because
every observation stream keys off the two fluxes; tests inspect the
conserved total separately via [`simulate_seir_daily_full`](@ref).
"""
function simulate_seir_daily(n::Integer;
        β::Real, σ::Real, γ::Real, CFR::Real, N::Real,
        S0::Real, E0::Real, I0::Real,
        R0::Real = zero(N), D0::Real = zero(N))
    Tp = promote_type(typeof(β), typeof(σ), typeof(γ),
                      typeof(CFR), typeof(S0), typeof(E0), typeof(I0))
    onsets = Vector{Tp}(undef, n)
    deaths = Vector{Tp}(undef, n)
    state  = (Tp(S0), Tp(E0), Tp(I0), Tp(R0), Tp(D0))
    @inbounds for t in 1:n
        state, new_onsets, new_deaths = step_seir_daily(state;
            β = β, σ = σ, γ = γ, CFR = CFR, N = N)
        onsets[t] = new_onsets
        deaths[t] = new_deaths
    end
    return (; onsets, deaths)
end

"""
$(TYPEDSIGNATURES)

Like [`simulate_seir_daily`](@ref) but also returns the full state
trajectory `(S, E, I, R, D)` over the `n` days. Used by tests to check
the conservation `S + E + I + R + D = N` and by validation against the
reference continuous-time solve.
"""
function simulate_seir_daily_full(n::Integer;
        β::Real, σ::Real, γ::Real, CFR::Real, N::Real,
        S0::Real, E0::Real, I0::Real,
        R0::Real = zero(N), D0::Real = zero(N))
    Tp = promote_type(typeof(β), typeof(σ), typeof(γ),
                      typeof(CFR), typeof(S0), typeof(E0), typeof(I0))
    S = Vector{Tp}(undef, n)
    E = Vector{Tp}(undef, n)
    I = Vector{Tp}(undef, n)
    R = Vector{Tp}(undef, n)
    D = Vector{Tp}(undef, n)
    onsets = Vector{Tp}(undef, n)
    deaths = Vector{Tp}(undef, n)
    state  = (Tp(S0), Tp(E0), Tp(I0), Tp(R0), Tp(D0))
    @inbounds for t in 1:n
        state, new_onsets, new_deaths = step_seir_daily(state;
            β = β, σ = σ, γ = γ, CFR = CFR, N = N)
        S[t], E[t], I[t], R[t], D[t] = state
        onsets[t] = new_onsets
        deaths[t] = new_deaths
    end
    return (; S, E, I, R, D, onsets, deaths)
end

## --- Delay discretisation (shared with the renewal candidate's idea) ----
##
## The owner's preferred route was CensoredDistributions.jl for the
## double-interval-censored delay PMFs. Its analytical primary-censored
## CDF for Gamma routes through `HypergeometricFunctions.pFqweniger`,
## which Mooncake cannot differentiate. We discretise here by trapezoidal
## quadrature of the density, with the value at zero forced to zero
## (exact for Gamma with shape > 1) so the `0·log 0` shape-parameter
## derivative is never evaluated. This is the same trick the package
## already uses in `ExportDeathDelay`; the bias from approximating the
## double-interval-censored PMF by a single-bin trapezoid is small
## relative to the prior uncertainty on the delay parameters and is
## documented in the proposal.

@inline function _safe_pdf(dist, x)
    z = zero(pdf(dist, oneunit(x)))
    return x <= zero(x) ? z : pdf(dist, x)
end

"""
$(TYPEDSIGNATURES)

Discretise a continuous delay `dist` to a daily PMF over lags
`0, 1, ..., nmax`. Each bin mass is the trapezoidal integral of the
density across `[d, d+1]`, then the vector is renormalised to sum to
one over the truncated support. AD-transparent under Mooncake: touches
the density only (never the Gamma CDF) and never evaluates the density
at zero.
"""
function discretise_delay_seir(dist, nmax::Integer)
    edges = 0:1:(nmax + 1)
    dens  = [_safe_pdf(dist, float(e)) for e in edges]
    pmf   = [(dens[i] + dens[i + 1]) / 2 for i in 1:(nmax + 1)]
    return pmf ./ sum(pmf)
end

"""
$(TYPEDSIGNATURES)

Discrete convolution of `x` with `delay` (indexed from lag 0): entry
`t` sums `x[t - d] · delay[d + 1]` over lags that stay in range.
Returns the expected daily counts of the delayed event on the same
grid.
"""
function convolve_delay_seir(x::AbstractVector, delay::AbstractVector)
    n = length(x)
    Tp = promote_type(eltype(x), eltype(delay))
    y = zeros(Tp, n)
    @inbounds for t in 1:n
        acc = zero(Tp)
        dmax = min(t - 1, length(delay) - 1)
        for d in 0:dmax
            acc += x[t - d] * delay[d + 1]
        end
        y[t] = acc
    end
    return y
end

## --- Safe NegBinomial -------------------------------------------------
##
## Same constructor the rest of the package uses. Duplicated here so the
## compartmental file is fully self-contained until the package-wide
## helper is promoted out of `docs/examples/analysis.jl` (issue #81).

"""
$(TYPEDSIGNATURES)

NegBinomial parameterised by mean `μ` and dispersion `k`, with a
NaN / Inf-safe `p` so an extreme NUTS proposal during warmup does not
trip the distribution's domain check.
"""
function safe_nbinomial_seir(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

## --- Submodels: one prior per rate / nuisance ---------------------------
##
## Every prior is its own submodel. To swap a prior (e.g. for sensitivity
## analysis), pass a customised submodel instance into `seir_growth_model`
## or the joint composer. No literal `Normal` / `Gamma` constants appear
## in observation model bodies; everything flows through these defaults.

"""
$(TYPEDSIGNATURES)

Basic reproduction number prior. Centred on `R0 = 2` with a 95% prior
interval roughly `[0.9, 4.4]`, matching the doubling-time prior used by
the exponential-growth model under the Euler-Lotka relation at the
prior-mean generation interval.
"""
@model function r0_model(;
        r0_prior = LogNormal(log(2.0), 0.4))
    R0 ~ r0_prior
    return (; R0)
end

"""
$(TYPEDSIGNATURES)

Latent-period prior `1/σ` (days from exposure to becoming infectious).
Truncated Normal centred at the BVD/Ebola incubation literature mean of
6 days with a 2-day SD. Returns the period and its reciprocal rate
`σ` so the stepper consumes either form.
"""
@model function latent_period_model(;
        latent_period_prior = truncated(Normal(6.0, 2.0); lower = 1.0))
    latent_period ~ latent_period_prior
    σ := 1.0 / latent_period
    return (; latent_period, σ)
end

"""
$(TYPEDSIGNATURES)

Infectious-period prior `1/γ` (days from onset to recovery or death).
Truncated Normal centred at 7 days with a 2-day SD, so the implicit
generation interval `1/σ + 1/γ` is roughly 13 days, matching the
renewal candidate's fixed gamma generation-interval mean.
"""
@model function infectious_period_model(;
        infectious_period_prior =
            truncated(Normal(7.0, 2.0); lower = 1.0))
    infectious_period ~ infectious_period_prior
    γ := 1.0 / infectious_period
    return (; infectious_period, γ)
end

"""
$(TYPEDSIGNATURES)

Seeding-cohort prior: a single zoonotic introduction parked in the
exposed compartment on day 1, with a small Normal⁺ around 1 so the
posterior can soften the dependence on an exact seed count.
"""
@model function seed_model(;
        seed_prior = truncated(Normal(1.0, 0.5); lower = 0.0))
    E0 ~ seed_prior
    return (; E0)
end

"""
$(TYPEDSIGNATURES)

Case-fatality ratio prior. `Beta(6.6, 13.4)` with mean ≈ 0.33 and 95%
roughly `[0.15, 0.54]`; same CDC-anchored prior the main model uses.
"""
@model function compartmental_cfr_model(;
        cfr_prior = Beta(6.6, 13.4))
    CFR ~ cfr_prior
    return (; CFR)
end

"""
$(TYPEDSIGNATURES)

Onset-to-death delay prior. Gamma`(α, θ)` with α and θ truncated
Normal priors anchored on the Isiro 2012 reanalysis. Returns the two
parameters and the constructed distribution.
"""
@model function compartmental_o2d_model(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0.0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0.0))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

"""
$(TYPEDSIGNATURES)

Onset-to-report delay prior. Gamma centred at the BVD field-report
window. Acts as a placeholder while the literature delay is refined.
"""
@model function compartmental_o2r_model(;
        alpha_prior = truncated(Normal(3.0, 1.0); lower = 0.0),
        theta_prior = truncated(Normal(2.0, 0.7); lower = 0.0))
    α_r ~ alpha_prior
    θ_r ~ theta_prior
    return (; α_r, θ_r, dist = Gamma(α_r, θ_r))
end

"""
$(TYPEDSIGNATURES)

Onset-to-detection delay prior (proxy for the detection window `w`).
Gamma with mean roughly 15 days, matching the prior on `w` used by the
existing Method 1 exports model.
"""
@model function compartmental_detect_model(;
        alpha_prior = truncated(Normal(4.0, 1.0); lower = 0.0),
        theta_prior = truncated(Normal(3.75, 1.0); lower = 0.0))
    α_d ~ alpha_prior
    θ_d ~ theta_prior
    return (; α_d, θ_d, dist = Gamma(α_d, θ_d))
end

"""
$(TYPEDSIGNATURES)

Daily-traveller-volume prior, defaulting to the same Normal⁺(1871, 200)
the main model uses.
"""
@model function compartmental_traveller_model(;
        mean::Real = ITURI_DAILY_TRAVEL,
        sd::Real   = ITURI_DAILY_TRAVEL_SD)
    daily_travellers ~ truncated(Normal(mean, sd); lower = 0.0)
    return (; daily_travellers)
end

"""
$(TYPEDSIGNATURES)

Surveillance dispersion prior on the `1/√k` scale, same form as the
main model. Returns both `inv_sqrt_k` and the implied `k`.
"""
@model function compartmental_dispersion_model(;
        inv_sqrt_k_prior = truncated(Normal(0.6, 0.2); lower = 0.0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

"""
$(TYPEDSIGNATURES)

Pooled-ascertainment prior, non-centred logit form, same as the main
model. Returns `(; p_drc, p_uganda, ...)`.
"""
@model function compartmental_ascertainment_model(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0.0, 0.5); lower = 1e-4))
    μ_logit  ~ mu_prior
    τ_logit  ~ tau_prior
    z_drc    ~ Normal(0.0, 1.0)
    z_uganda ~ Normal(0.0, 1.0)
    logit_p_drc    = μ_logit + τ_logit * z_drc
    logit_p_uganda = μ_logit + τ_logit * z_uganda
    p_drc    := logistic(logit_p_drc)
    p_uganda := logistic(logit_p_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

## --- Growth submodel ----------------------------------------------------

"""
$(TYPEDSIGNATURES)

Compartmental growth submodel: composes the rate-and-seed priors and
runs the daily SEIRD stepper for `ceil(m·τ)` days, where `τ` is the
doubling time implied by the Euler-Lotka relation at the sampled
`(R0, σ, γ)`. Returns a NamedTuple carrying the onset trajectory and
summary scalars:

- `T`            outbreak age in days (real-valued, computed from
                 `m·τ`; the daily grid uses `ceil`).
- `r`            implied early growth rate.
- `τ`            implied doubling time `log(2) / r`.
- `m`            doubling-times-to-seeding (sampled).
- `C_T`          expected cumulative onsets by `T`.
- `onset_daily`  the length-`n` daily onset vector — the seam used by
                 every observation stream.
- `R0, σ, γ, β, CFR, E0, latent_period, infectious_period` — sampled
                 latent-process parameters, exposed so the joint
                 composer / posterior summaries can read them off
                 `growth_state` directly.

Every prior is its own submodel; pass a customised instance into the
kwargs to override.
"""
@model function seir_growth_model(;
        N::Real     = float(ITURI_POPULATION),
        r0          = r0_model(),
        latent      = latent_period_model(),
        infectious  = infectious_period_model(),
        seed        = seed_model(),
        cfr         = compartmental_cfr_model(),
        m_prior     = truncated(Normal(7.0, 2.5);
                                 lower = 0.0, upper = 13.0))
    r0_state         ~ to_submodel(r0, false)
    latent_state     ~ to_submodel(latent, false)
    infectious_state ~ to_submodel(infectious, false)
    seed_state       ~ to_submodel(seed, false)
    cfr_state        ~ to_submodel(cfr, false)

    R0  = r0_state.R0
    σ   = latent_state.σ
    γ   = infectious_state.γ
    E0  = seed_state.E0
    CFR = cfr_state.CFR

    # β set from R0 via β = R0 · γ (one infectious individual produces
    # R0 new infections over 1/γ days at S ≈ N).
    β = R0 * γ
    # Early growth rate from the Euler-Lotka relation for a sum-of-two-
    # exponentials generation interval: (1 + r/σ)(1 + r/γ) = R0. Closed
    # form via the quadratic in r: r² + (σ + γ) r + σγ(1 − R0) = 0.
    a_coef = σ + γ
    b_coef = σ * γ * (one(R0) - R0)
    disc   = a_coef^2 - 4 * b_coef
    r      = (-a_coef + sqrt(_seir_safe_pos(disc))) / 2

    τ := log(2.0) / _seir_safe_pos(r)

    m ~ m_prior
    T_real = m * τ
    n = max(1, ceil(Int, T_real))

    sim = simulate_seir_daily(n;
        β = β, σ = σ, γ = γ, CFR = CFR, N = N,
        S0 = N - E0, E0 = E0, I0 = zero(N))
    onset_daily = sim.onsets

    C_T := sum(onset_daily)
    T   := float(T_real)

    return (; τ = τ, r = r, m = m, T = T_real, C_T = sum(onset_daily),
              onset_daily = onset_daily,
              R0 = R0, σ = σ, γ = γ, β = β,
              latent_period = latent_state.latent_period,
              infectious_period = infectious_state.infectious_period,
              CFR = CFR, E0 = E0, N = N)
end

## --- Onset-staged observation submodels --------------------------------

"""
$(TYPEDSIGNATURES)

Deaths likelihood for the compartmental model. Expected daily deaths
are the onset trajectory convolved with the onset-to-death PMF
(discretised once per draw, AD-transparent under Mooncake) and scaled
by `CFR`; the cumulative expectation at the cut-off is the observation
mean. NegBinomial likelihood with dispersion `k`. `dist_factory`
defaults to [`safe_nbinomial_seir`](@ref); pass another factory to
swap in (e.g. Poisson) the observation distribution.
"""
@model function deaths_onset_model(
        total_deaths::Union{Missing, Integer},
        growth_state, k::Real;
        o2d           = compartmental_o2d_model(),
        dist_factory  = safe_nbinomial_seir,
        delay_max::Integer = 60)
    o2d_state ~ to_submodel(o2d, false)
    pmf = discretise_delay_seir(o2d_state.dist, delay_max)
    daily_deaths = convolve_delay_seir(growth_state.onset_daily, pmf) .*
        growth_state.CFR
    expected_deaths_T := _seir_safe_pos(sum(daily_deaths))
    total_deaths ~ dist_factory(k, expected_deaths_T)
    return (; expected_deaths_T, o2d_state)
end

"""
$(TYPEDSIGNATURES)

Reported-cases likelihood. Expected daily reports are the onset
trajectory convolved with the onset-to-report PMF, scaled by the DRC
ascertainment fraction `p_drc`. NegBinomial likelihood with shared
dispersion `k`.
"""
@model function cases_onset_model(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real;
        o2r           = compartmental_o2r_model(),
        dist_factory  = safe_nbinomial_seir,
        delay_max::Integer = 30)
    o2r_state ~ to_submodel(o2r, false)
    pmf = discretise_delay_seir(o2r_state.dist, delay_max)
    daily_reports = convolve_delay_seir(growth_state.onset_daily, pmf) .*
        p_drc
    expected_reports_T := _seir_safe_pos(sum(daily_reports))
    reported_cases ~ dist_factory(k, expected_reports_T)
    return (; expected_reports_T, o2r_state)
end

"""
$(TYPEDSIGNATURES)

Exports likelihood. Expected daily detected exports are the onset
trajectory convolved with the onset-to-detection PMF, scaled by the
travel rate `q = daily_travellers / source_population` and the Uganda
ascertainment fraction `p_uganda`. Poisson likelihood (two
observations cannot identify a separate dispersion).
"""
@model function exports_onset_model(
        exported_cases::Union{Missing, Integer},
        growth_state, p_uganda::Real;
        detect        = compartmental_detect_model(),
        traveller     = compartmental_traveller_model(),
        dist_factory  = Poisson,
        source_population::Real = ITURI_POPULATION,
        delay_max::Integer = 45)
    detect_state    ~ to_submodel(detect, false)
    traveller_state ~ to_submodel(traveller, false)

    pmf = discretise_delay_seir(detect_state.dist, delay_max)
    q   = traveller_state.daily_travellers / source_population
    daily_exports = convolve_delay_seir(growth_state.onset_daily, pmf) .*
        (p_uganda * q)
    expected_exports_T := _seir_safe_pos(sum(daily_exports))
    exported_cases ~ dist_factory(expected_exports_T)
    return (; expected_exports_T, detect_state, traveller_state)
end

## --- Joint composer ----------------------------------------------------

"""
$(TYPEDSIGNATURES)

Compartmental joint composer. Combines [`seir_growth_model`](@ref)
with onset-staged exports / deaths / cases likelihoods (all priors
injected via the submodel kwargs). Each stream argument may be
`missing`, so the composer doubles as a prior generator. The TMRCA
soft bound is injected via `genetic` analogously to `bvd_joint`.
"""
@model function bvd_compartmental_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer} = missing;
        growth        = seir_growth_model(),
        exports       = exports_onset_model,
        deaths        = deaths_onset_model,
        cases         = cases_onset_model,
        dispersion    = compartmental_dispersion_model(),
        ascertainment = compartmental_ascertainment_model(),
        genetic       = nothing,
        source_population::Real = ITURI_POPULATION)

    growth_state     ~ to_submodel(growth, false)
    if genetic !== nothing
        genetic_state ~ to_submodel(genetic(growth_state.T), false)
    end
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)

    k        = dispersion_state.k
    p_drc    = asc_state.p_drc
    p_uganda = asc_state.p_uganda

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state, p_uganda;
                source_population = source_population), false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)
    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k, p_drc), false)

    cumulative_cases := growth_state.C_T
end
