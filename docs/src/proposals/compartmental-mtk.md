# Proposal: composable compartmental architecture (Catalyst.jl)

Status: candidate redesign for review, branch `arch-compartmental-mtk`.
Author: drafted by an agent under human oversight.

## Motivation

The current model treats the latent epidemic as a deterministic continuous-time cumulative-incidence trajectory $C(s) = e^{rs}$.
A renewal alternative (`arch-renewal`) replaces $r$ with a daily renewal recursion driven by $R_t$.
This proposal adds a third option: a **compartmental (SEIR) latent process** defined as a Catalyst.jl reaction network.

The motivation is composability.
With a Catalyst reaction network the compartmental specification is a first-class symbolic object: each compartment and each flow is a symbolic entity that can be extended (add a hospitalisation compartment, branch I into ascertained / unascertained, add a vaccination flow) without rewriting the forward map.
This contrasts with the renewal candidate, which is one hand-rolled loop, and with the continuous-time explicit-convolution candidate, where the forward operator is the convolution itself.

## Choice of framework

We use **Catalyst.jl** as the front end and a hand-rolled discrete-time stepper as the back end, with continuous-time ODE solves available for reference.
The reasoning:

- Catalyst's `@reaction_network` DSL is the natural fit for SEIR flows.
  Each `S + I --> E + I` arrow is a first-class symbolic reaction; new compartments compose by adding arrows, not by hand-editing a vector field.
- ModelingToolkit.jl wraps the network and emits the symbolic vector field, the Jacobian, and generated code.
  We use it as the symbolic backbone and read the rates / stoichiometries out at construction time.
- For the forward map, the runtime cost / AD risk of routing every NUTS proposal through `OrdinaryDiffEq.solve` is non-trivial.
  Mooncake's support for `OrdinaryDiffEq` is limited (see the AD verdict below).
  A daily semi-implicit Euler step is sufficient at our daily-data resolution and integrates cleanly under Mooncake.

The runtime path is therefore: Catalyst defines the network symbolically once at module load, a daily stepper consumes the rate-law / stoichiometry information, and the Turing model differentiates through the stepper alone.

## Latent process

Compartments: $S, E, I, R, D$ (susceptible, exposed, infectious, recovered, dead).
Catalyst reactions:

$$
S + I \xrightarrow{\beta / N} E + I,\qquad
E \xrightarrow{\sigma} I,\qquad
I \xrightarrow{(1-\mathrm{CFR})\gamma} R,\qquad
I \xrightarrow{\mathrm{CFR}\,\gamma} D.
$$

Submodel priors (one per rate, all delays via priors, as required by the design rules):

- $1/\sigma$: latent period (E residence time), $\mathrm{Normal}^+(\mu = 6,\ \sigma = 2)$ days (Ebola/BVD incubation literature).
- $1/\gamma$: infectious period (I residence time), $\mathrm{Normal}^+(\mu = 7,\ \sigma = 2)$ days.
- $\mathcal R_0 = \beta / \gamma$ rather than $\beta$ directly: $\mathcal R_0 \sim \mathrm{LogNormal}(\log 2, 0.4)$, so the implicit generation interval is $1/\sigma + 1/\gamma$ and the early-growth rate is recovered via the Euler-Lotka relation rather than fixed by hand.
- $\mathrm{CFR}$: shared with the rest of the package (Beta(6.6, 13.4)).
- Seeding: $E(0) = 1, I(0) = 0$, $S(0) = N - 1$ where $N$ is the source population (fixed).
- Outbreak age $T$: priors $m \sim \mathrm{Normal}^+(7, 2.5)$ doubling times until $T$, so $T = m \tau$ with $\tau$ derived from $(\mathcal R_0, \sigma, \gamma)$ via Euler-Lotka — the same prior interface as the exponential-growth submodel, so the joint composer can swap in this submodel by interface alone.

### S-depletion verdict

At the prior-mean $\mathcal R_0 \approx 2$, doubling time $\tau \approx 14$ d, and $T = m\tau \approx 100$ d, the expected $C(T)$ is $\sim 10^2$ - $10^3$ against $N \approx 4.4 \times 10^6$.
$S$ depletion is therefore $\lesssim 10^{-3}$ across the prior, so the SEIR trajectory is indistinguishable from the linear-cumulative exponential approximation to the same precision the data can resolve.
This is not a reason against the compartmental architecture — its value is the composable scaffolding for richer structure later (heterogeneous mixing, intervention compartments, vaccination), not better fits to the present aggregate data.

## From compartments to onsets

The seam to the observation streams is **onsets**, defined as the daily $E \to I$ flux:
$$
\mathrm{onset}_t \equiv \sigma\, E_{t}.
$$

This is computed once per draw and reused across all four observation streams.
Cumulative cases is the running sum of onsets (equivalent to $N - S(t)$ in the negligible-depletion limit).

## How the four streams plus TMRCA map in

| Stream | Mapping in the compartmental model |
|---|---|
| 1. Exports in Uganda | $p_{\mathrm{Uganda}}\, q \sum_t (\mathrm{onset} * F_{\mathrm{det}})_t$ with $F_{\mathrm{det}}$ the onset-to-detection survival window, Poisson |
| 2. DRC suspected deaths | $\mathrm{CFR}\sum_t (\mathrm{onset} * f_{\mathrm{o2d}})_t$, NegBinomial$(k)$ |
| 3. DRC reported cases | $p_{\mathrm{DRC}}\sum_t (\mathrm{onset} * f_{\mathrm{o2r}})_t$, NegBinomial$(k)$ |
| 4. Deaths among exports | $\mathrm{CFR}\, p_{\mathrm{Uganda}}\, q \sum_t (\mathrm{onset} * F_{\mathrm{det}} * f_{\mathrm{o2d}})_t$, Poisson |
| TMRCA soft bound | $\Phi((T - g)/\sigma_{\mathrm{tmrca}})$ on the log density |

The cumulative on $[T-w, T]$ that the exports stream needs is exactly $\sum_{t=T-w+1}^{T} \mathrm{onset}_t$ in this discretised form, which is what the continuous-conv candidate evaluates by quadrature.

## Discrete-time forward map (the stepper)

For each day $t = 1, \dots, n$ ($n = T$ on the daily grid), a semi-implicit Euler step on the Catalyst reactions:

1. $\lambda_t = \beta\, I_{t} / N$  (force of infection).
2. New infections $\Delta_t^{SE} = (1 - e^{-\lambda_t})\, S_t$.
3. New onsets $\Delta_t^{EI} = (1 - e^{-\sigma})\, E_t$.
4. New removals $\Delta_t^{I\cdot} = (1 - e^{-\gamma})\, I_t$, branched into deaths $(\mathrm{CFR})$ and recoveries $(1 - \mathrm{CFR})$.
5. $S_{t+1} = S_t - \Delta^{SE}$, etc.

The $(1 - e^{-\mathrm{rate}})$ form is the exact transition probability for a constant-rate exponential clock over a unit interval; it preserves $S + E + I + R + D = N$ exactly and degenerates correctly when rates are small.
This is the standard "exponentialised-rate" daily discretisation.
It is AD-transparent: only `exp`, multiplication, subtraction, with no division by latent quantities and no calls into solver internals.

## AD route — Mooncake verdict

We probed `Mooncake.value_and_gradient` on a closure that builds an `ODEProblem` from a Catalyst SEIR reaction network and calls `OrdinaryDiffEq.solve(prob, Tsit5())`.
The forward solve succeeded; the Mooncake gradient build failed with `TypeError: in typeassert, expected Mooncake.CoDual{IdDict{Any, Any}, IdDict{Any, Any}}, got a value of type Mooncake.CoDual{IdDict{Any, Any}, Mooncake.NoFData}`, originating in the SciML internals that thread `IdDict` caches.
`SciMLSensitivity` adjoints route through ChainRules / Zygote rather than Mooncake, so the adjoint path does not rescue this either.

The verdict is therefore to **not route NUTS through the ODE solver**, and instead differentiate through a daily semi-implicit Euler stepper.
The stepper is a pure-Julia loop using only AD-friendly primitives (multiplication, subtraction, `exp` of bounded quantities) and Mooncake handles it cleanly — the same construction the renewal candidate already uses for its daily recursion.
The continuous ODE path is retained in the package as a reference forward evaluator (for prior-predictive plots and validation) but not used for inference.

The headline runtime cost vs the current model: each draw evaluates an $n$-step loop instead of a Gauss-Legendre quadrature.
The renewal candidate shows this stays competitive on the same hardware ($n \le 200$ days, daily grid).

## Identifiability risks

The compartmental specification introduces three rate parameters ($\mathcal R_0$, $\sigma$, $\gamma$) where the exponential-growth model has one ($\tau$).
With four aggregate counts the data cannot identify all three; they will lean on their priors.
We mitigate this by:

- Parameterising in $\mathcal R_0$ rather than $\beta$, so the prior is on the textbook quantity and inherits its prior interval directly.
- Sampling the latent period $1/\sigma$ and infectious period $1/\gamma$ from informative priors anchored on the BVD/Ebola literature, with the joint $1/\sigma + 1/\gamma$ matching the renewal candidate's fixed generation-interval mean.
- Reporting the implied $\tau$ as a derived quantity so the posterior can be compared directly with the exponential-growth fit (sanity check).

The composable upside — being able to extend the model with a hospitalisation compartment, a vaccinated stratum, or time-varying $\beta_t$ — only becomes valuable when more granular data arrives.
For the current aggregate-only data, the joint posterior on $C(T)$ should track the renewal posterior closely, modulo the additional prior structure on the residence times.

## Drift-from-replication assessment

The compartmental model is a fundamentally different latent process from the McCabe et al. exponential.
Three drift dimensions:

1. **S-depletion**: negligible at this outbreak size, as above.
2. **Generation-interval implicit, not explicit**: the Euler-Lotka relation $r = $ root of $\int_0^\infty e^{-rs} g(s) ds = 1/\mathcal R_0$ with $g$ the sum of two exponentials replaces the renewal model's prescribed gamma generation interval. The two are not identical: a sum-of-two-exponentials generation interval is a hypoexponential, not a gamma.
3. **Onset definition**: $\sigma E$ vs the renewal model's lagged $I$ convolution. These coincide in the steady-growth regime but the compartmental version is internally consistent across regime changes.

Drift from McCabe et al.'s exact formulae is therefore expected and should be reported alongside the existing comparison table, not papered over.

## Expected runtime

The renewal candidate completes a 1000-draw, 4-chain NUTS fit in roughly the same wall-clock as the current model ($\sim 6$ min on the developer machine).
The compartmental stepper is the same structure (one daily loop per draw) plus a few extra multiplications per step, so we expect comparable runtime, possibly marginally slower.
A full smoke fit will be reported in the PR.

## Modularity (per the design rules)

Submodels (each prior owned by one):

- `r0_model()` — $\mathcal R_0$ prior.
- `latent_period_model()` — $1/\sigma$ prior.
- `infectious_period_model()` — $1/\gamma$ prior.
- `seir_growth_model()` — composes the three above, runs the daily stepper, returns `(; onset_daily, cumulative_daily, T, C_T, r, cumulative)` so it can plug into the existing observation submodels by interface.
- All four observation streams reuse the existing onset-staged path (`exports_model`, `deaths_model`, `cases_model`, `exports_deaths_model`) unchanged — the seam is `onset_daily`.

No inline `using` in `compartmental.jl` (only the enclosing module imports).
No literal `Normal`/`Gamma` constants buried in model bodies (every prior is a submodel default).
Observation distributions are injected via the composer interface, as in the current package.

## Deliverable scope

This branch delivers:

1. This proposal.
2. `src/compartmental.jl` as proper package code, MODULAR, alongside the existing exponential-growth model.
3. A demonstrator that prior-predictive draws run, Mooncake gradients succeed through the chosen route, and a short NUTS smoke fit completes.
4. New tests asserting the stepper's mass-conservation and the AD path's finiteness.

Not in scope: a full posterior fit, posterior-predictive figures, or an analysis-page entry.
Those follow once the architecture is reviewed.
