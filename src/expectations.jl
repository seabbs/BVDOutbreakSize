# Expected-count integrals built on top of the shared `integrate`
# helpers: expected cumulative deaths, expected detected exports, and
# expected deaths-among-exports.

"""
Expected cumulative deaths by time `T` from a single seeding case under
exponential growth at rate `r`:

```math
\\mathbb{E}[D_T] = \\mathrm{CFR} \\cdot
    \\int_0^T e^{r s}\\, f(T - s)\\, ds,
```

with `f` the `delay_dist` onset-to-death density. The integrand returns
zero past `T` so the convolution support is respected. Integrated with
the clustered [`integrate`](@ref) so the quadrature nodes pack near `T`,
where `f(T − s)` has mass, over a window set by the delay scale (see
[`DELAY_SUPPORT_K`](@ref)). Uses [`DEATH_INTEGRAL_ALG`](@ref).
"""
function expected_deaths(CFR, r, T, delay_dist; alg = DEATH_INTEGRAL_ALG)
    scale = _delay_scale(delay_dist)
    g = let r = r, T = T, delay_dist = delay_dist
        s -> begin
            d = T - s
            d <= 0 ? zero(r) : exp(r * s) * pdf(delay_dist, d)
        end
    end
    return CFR * integrate(g, zero(T), T, scale; alg)
end

"""
Expected cumulative deaths by time `T` from a single seeding case under
exponential growth at rate `r`, using analytic result specific to the
Gamma distribution:

```math
\\mathbb{E}[D_T] = \\mathrm{CFR} \\cdot
    \\int_0^T e^{r s}\\, f(T - s)\\, ds = \\mathrm{CFR} \\cdot e^{r T} \\cdot M(-r) \\cdot F(T (1 + \\theta r)),
```

where f is the Gamma(α, θ) density, M is its moment-generating function,
and F is its CDF. The CDF is routed through [`_gamma_cdf`](@ref) so
Mooncake reverse-mode AD picks up a shape-parameter gradient (computed
internally with ForwardDiff).
"""
function expected_deaths(CFR, r, T, delay_dist::Gamma)
    α, θ = delay_dist.α, delay_dist.θ
    return CFR * exp(r * T) * mgf(delay_dist, -r) *
           _gamma_cdf(α, θ, T * (1 + θ * r))
end

"""
Expected cumulative detected exports by elapsed time `t`, clamped to be
strictly positive and finite:

```math
\\mathbb{E}[\\text{exports}(t)] = p \\cdot q \\cdot
    \\int_{t-w}^{t} C(s)\\, ds,
```

with `cumulative` the trajectory ``C(s)``, `p` the detection
probability, `q` the per-capita travel rate, and `window` the detection
window ``w``. Backs both the exports count likelihood (evaluated at
`t = T`) and the first-export-detection survival term (evaluated at an
earlier `t`). Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_exports(cumulative, p, q, t, window;
        alg = CUMULATIVE_INTEGRAL_ALG)
    window_start = max(t - window, zero(t))
    integral = integrate_cumulative(cumulative, window_start, t; alg)
    raw = p * q * integral
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end

"""
Expected cumulative deaths among detected exports by elapsed time `t`,
clamped to be strictly positive and finite:

```math
\\mathbb{E}[D_{\\text{uganda}}(t)] = \\mathrm{CFR} \\cdot
    p \\cdot q \\cdot
    \\int_{t-w}^{t} C(s)\\, F_d(t - s)\\, ds,
```

with `cumulative` the trajectory ``C(s)``, `delay_dist` the onset-to-death
distribution (CDF ``F_d``), `p` the detection probability, `q` the
per-capita travel rate, and `window` the detection window ``w``. Backs
the binned-Poisson export-death likelihood, evaluated at the daily bin
edges. Uses [`CUMULATIVE_INTEGRAL_ALG`](@ref).
"""
function expected_exports_deaths(cumulative, delay_dist, CFR, p, q,
        t, window; alg = CUMULATIVE_INTEGRAL_ALG)
    window_start = max(t - window, zero(t))
    integral = integrate_exports_deaths(
        cumulative, delay_dist, window_start, t, t; alg)
    raw = CFR * p * q * integral
    return isfinite(raw) ? max(raw, eps(typeof(raw))) : eps(typeof(raw))
end
