# Scratch benchmark: speed up the deaths-among-exports nested integral.
#
# Baseline `integrate_exports_deaths(cumulative, delay_dist, …)` is a
# 32-node outer Gauss-Legendre integral whose integrand recomputes a
# 32-node inner CDF integral at every outer node (~1024 pdf evals per
# call). The model loop in `exports_deaths_model` calls it once per daily
# bin edge, and every call is differentiated through with Mooncake.
#
# This compares the shipped fast path — `ExportDeathDelay`, which
# precomputes `F_d` once on a grid and interpolates — against the
# distribution method (the reference) over a realistic loop of bin edges,
# reporting primal time, Mooncake gradient time and accuracy. A second,
# value-only candidate (vectorised inner CDF, no IntegralProblem per node)
# is included to show the intermediate option discussed in the PR.
#
# Run: julia --project scripts/experiment_integral_speed.jl

using BVDOutbreakSize: integrate, integrate_exports_deaths,
                       ExportDeathDelay, CUMULATIVE_INTEGRAL_ALG
using Distributions: Gamma, pdf
using Integrals: IntegralProblem, QuadGKJL, solve
import FastGaussQuadrature
using Mooncake: Mooncake
using Printf: @printf

const ALG = CUMULATIVE_INTEGRAL_ALG          # GaussLegendre(n = 32)
const NQ  = 32

# --- candidate A (value-only): vectorised inner CDF --------------------
# One shared inner Gauss-Legendre node set for all outer nodes; same flop
# count as the baseline but no per-node IntegralProblem/solve. Exact.
const _GLN, _GLW = FastGaussQuadrature.gausslegendre(NQ)
const _V01 = (_GLN .+ 1) ./ 2
const _W01 = _GLW ./ 2
function xd_vectorised(cumulative, delay_dist, lo, hi, T)
    hi <= lo && return zero(hi - lo)
    halfwidth = (hi - lo) / 2
    acc = zero(promote_type(typeof(lo), typeof(T)))
    @inbounds for j in 1:NQ
        s, y = halfwidth * (_GLN[j] + 1) + lo, zero(T)
        y = T - (halfwidth * (_GLN[j] + 1) + lo)
        Fd = zero(y)
        if y > zero(y)
            inner = zero(y)
            for k in 1:NQ
                inner += _W01[k] * pdf(delay_dist, y * _V01[k])
            end
            Fd = y * inner
        end
        acc += _GLW[j] * cumulative(halfwidth * (_GLN[j] + 1) + lo) * Fd
    end
    return halfwidth * acc
end

# --- adaptive reference (double integral) ------------------------------
function xd_reference(cumulative, delay_dist, lo, hi, T)
    cdf_to(x) = x <= 0 ? 0.0 :
        solve(IntegralProblem((u, p) -> pdf(delay_dist, u), (0.0, x), nothing),
              QuadGKJL(); reltol = 1e-12, abstol = 1e-14).u
    g = (s, p) -> cumulative(s) * cdf_to(T - s)
    return solve(IntegralProblem(g, (lo, hi), nothing), QuadGKJL();
                 reltol = 1e-10, abstol = 1e-12).u
end

# --- realistic loop: n+1 bin edges, exponential growth -----------------
const R0, A0, TH0 = log(2) / 14, 4.3, 2.6
const T0, W0, N0  = 90.0, 20.0, 4

function loop_baseline(r, α, θ, T, window, n)
    d, C, tot = Gamma(α, θ), s -> exp(r * s), zero(r)
    for i in 0:n
        t = T - n + i; lo = max(t - window, zero(t))
        tot += integrate_exports_deaths(C, d, lo, t, t)
    end
    return tot
end
function loop_vector(r, α, θ, T, window, n)
    d, C, tot = Gamma(α, θ), s -> exp(r * s), zero(r)
    for i in 0:n
        t = T - n + i; lo = max(t - window, zero(t))
        tot += xd_vectorised(C, d, lo, t, t)
    end
    return tot
end
function loop_grid(r, α, θ, T, window, n)
    ed  = ExportDeathDelay(Gamma(α, θ), window)
    C, tot = s -> exp(r * s), zero(r)
    for i in 0:n
        t = T - n + i; lo = max(t - window, zero(t))
        tot += integrate_exports_deaths(C, ed, lo, t, t)
    end
    return tot
end

function bench(f, reps)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end
gb(x) = loop_baseline(x[1], x[2], x[3], x[4], W0, N0)
gv(x) = loop_vector(x[1], x[2], x[3], x[4], W0, N0)
gg(x) = loop_grid(x[1], x[2], x[3], x[4], W0, N0)

function grad_time(f, x, reps)
    cache = Mooncake.prepare_gradient_cache(f, x)
    Mooncake.value_and_gradient!!(cache, f, x)
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed Mooncake.value_and_gradient!!(cache, f, x))
    end
    _, gs = Mooncake.value_and_gradient!!(cache, f, x)
    return t, gs[2]
end

function main()
    println("=== Loop sum accuracy (n+1 = $(N0 + 1) bin edges) ===")
    refsum = let C = s -> exp(R0 * s), d = Gamma(A0, TH0)
        s = 0.0
        for i in 0:N0
            t = T0 - N0 + i; l = max(t - W0, 0.0)
            s += xd_reference(C, d, l, t, t)
        end; s
    end
    b = loop_baseline(R0, A0, TH0, T0, W0, N0)
    v = loop_vector(R0, A0, TH0, T0, W0, N0)
    g = loop_grid(R0, A0, TH0, T0, W0, N0)
    for (name, x) in (("baseline (dist)", b), ("vectorised     ", v),
                      ("ExportDeathDelay", g))
        @printf("  %s  relerr_vs_ref=%.2e  relerr_vs_baseline=%.2e\n",
                name, abs(x - refsum) / abs(refsum), abs(x - b) / abs(b))
    end

    println("\n=== Primal time (loop sum, best of 50) ===")
    tb = bench(() -> loop_baseline(R0, A0, TH0, T0, W0, N0), 50)
    tv = bench(() -> loop_vector(R0, A0, TH0, T0, W0, N0), 50)
    tg = bench(() -> loop_grid(R0, A0, TH0, T0, W0, N0), 50)
    @printf("  baseline (dist)   %.2f µs   (1.00x)\n", tb * 1e6)
    @printf("  vectorised        %.2f µs   (%.2fx)\n", tv * 1e6, tb / tv)
    @printf("  ExportDeathDelay  %.2f µs   (%.2fx)\n", tg * 1e6, tb / tg)

    println("\n=== Mooncake gradient time (best of 20) ===")
    x = [R0, A0, TH0, T0]
    gtb, db = grad_time(gb, x, 20)
    gtv, dv = grad_time(gv, x, 20)
    gtg, dg = grad_time(gg, x, 20)
    @printf("  baseline (dist)   %.2f µs   (1.00x)\n", gtb * 1e6)
    @printf("  vectorised        %.2f µs   (%.2fx)\n", gtv * 1e6, gtb / gtv)
    @printf("  ExportDeathDelay  %.2f µs   (%.2fx)\n", gtg * 1e6, gtb / gtg)
    @printf("  grad finite: vectorised=%s  ExportDeathDelay=%s\n",
            all(isfinite, dv), all(isfinite, dg))
    @printf("  grad relerr vs baseline: vectorised=%.2e  grid=%.2e\n",
            maximum(abs.(dv .- db) ./ max.(abs.(db), 1e-30)),
            maximum(abs.(dg .- db) ./ max.(abs.(db), 1e-30)))
end

main()
