# # Estimating the current size of the 2026 DRC Bundibugyo virus outbreak: a joint Bayesian re-analysis of the McCabe et al. report
#
# **Authors.** Sam Abbott, Kath Sherratt, Samuel Brand and Sebastian
# Funk.
#
#md # ```@eval
#md # using Dates, Markdown
#md # Markdown.parse("**Last updated.** $(Dates.today()). This is a " *
#md #     "live report, re-run as new data arrive, so the estimates " *
#md #     "change between updates.")
#md # ```
#
#md # ```@eval
#md # using BVDOutbreakSize, Markdown
#md # d = load_observations().as_of_date
#md # Markdown.parse("**Data as of.** $(d), the release date of the " *
#md #     "WHO AFRO External Situation Report 01 the counts are taken " *
#md #     "from. Estimates are reported as of this date; it can lag " *
#md #     "the update date above.")
#md # ```
#
#md # ```@eval
#md # using BVDOutbreakSize, Markdown
#md # readme = read(joinpath(pkgdir(BVDOutbreakSize), "README.md"), String)
#md # m = match(r"<!-- ABSTRACT:START -->(.*?)<!-- ABSTRACT:END -->"s, readme)
#md # Markdown.parse(strip(m.captures[1]))
#md # ```
#
# **Use of AI.** The model code and this analysis were drafted by a
# language model and reviewed and revised under human oversight; the
# named authors are responsible for that oversight (see the
# *LLM-driven reimplementation* limitation below). The full analysis
# code lives in the
# [epiforecasts/BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize)
# repository, where issues and suggestions are welcome. This page is
# generated from
# [`docs/examples/analysis.jl`](https://github.com/epiforecasts/BVDOutbreakSize/blob/main/docs/examples/analysis.jl);
# the model code it calls is in
# [`src/`](https://github.com/epiforecasts/BVDOutbreakSize/tree/main/src).
#
# **How the numbers differ from McCabe et al.** Our estimates differ
# from the McCabe et al. [mccabe2026](@cite) report for two reasons.
# First, the method: we fit
# all streams jointly in a single Bayesian model rather than combining
# separate scenario analyses (see
# [What we do differently](#What-we-do-differently-from-McCabe-et-al.)
# below). Second, the data: results are reported as of the cut-off
# date in `data/observations.toml` (currently **2026-05-18**), using
# the reported counts in the data table below. These match the 20 May
# report's deaths and cases ($131$ and $516$) and are more recent than
# the 16 May 2026 figures the 18 May report used (e.g. $88$ suspected
# deaths). The joint
# posterior assumes a single common
# cut-off for every data stream, so the deaths, exports and reported-
# case counts must all be kept in sync to the same date.
#
# **Offline copy.** A self-contained single-file HTML version of this
# report, built from the same run, is attached to each results release:
# [download the latest](https://github.com/epiforecasts/BVDOutbreakSize/releases/latest/download/analysis.html).
#
# **→ Jump straight to the results:**
# [current outbreak size](#Results) ·
# [comparison with McCabe et al.](#Comparison-with-McCabe-et-al.) ·
# [how the data streams compare](#How-the-data-streams-compare) ·
# [limitations](#Limitations).
#
# ## What we do differently from McCabe et al.
#
# - *Joint posterior, not 15 scenario estimates.* The doubling time
#   $\tau$, case fatality ratio (CFR), onset-to-death shape and scale,
#   detection window $w$,
#   daily traveller volume and surveillance dispersion all have
#   priors and are sampled jointly. McCabe et al.
#   [mccabe2026](@cite) fix each and
#   sweep.
# - *Exact cumulative integral for exports* —
#   $q\cdot\int_{T-w}^{T} C(s)\,ds$ with travel rate $q$ — rather than
#   the small-$rw$ simplification $q\cdot w\cdot C(T)$ used by McCabe et
#   al. (and by [imai2020](@cite) before them). The two forms agree
#   as $r \to 0$.
# - *Numerical (not closed-form) deaths convolution.* For a gamma
#   delay the convolution integral has an exact closed form that
#   carries a $\gamma(\alpha, (\beta + r)T)/\Gamma(\alpha)$ factor from
#   the finite upper limit $T$. McCabe et al. use the large-$T$
#   simplification
#   $D(T) \approx \mathrm{CFR}\cdot C(T)\cdot(1 + r/\beta)^{-\alpha}$,
#   which drops that factor and
#   is therefore an approximation. We evaluate the integral
#   numerically instead, which recovers the exact value and lets the
#   onset-to-death distribution be swapped for any other family.
# - *Onset-to-death prior anchored on the Bayesian reanalysis* of
#   the same Isiro 2012 line list McCabe et al. cite for their
#   point estimates [bdbv_linelist_analysis_2026](@cite),
#   so the priors carry the published 95% credible intervals on
#   $\alpha$ and $\theta$ rather than collapsing onto Rosello's point
#   estimate.
# - *NegBinomial likelihood on deaths and reported cases* with a
#   single shared surveillance dispersion $k$. McCabe et al. use
#   Poisson for deaths and do not have a cases-ascertainment
#   model at all. Exports stay Poisson because two observations
#   would not identify a separate dispersion. The McCabe et al.
#   "exact NegBinomial CIs" on Method 1 are the conventional
#   binomial-inversion procedure, not an estimated dispersion.
# - *Ascertainment extension* (not in McCabe et al.). A logit-scale
#   hyperprior on the reporting fraction, applied to the latent
#   $C(T)$, gives a joint posterior over the reported suspected-case
#   count alongside deaths and exports.
# - *No-onward-transmission counterfactual* (not in McCabe et al.).
#   Projects the future expected deaths from cases already infected
#   by $T$, integrating $i(s)\cdot(1 - F_d(T - s))$ per draw — a
#   lower bound on the eventual death toll if every onward
#   transmission stopped today.
#
# ## Limitations
#
# - *Fitted only to aggregate reported counts.* The data are a
#   handful of summary figures — total suspected cases in the DRC,
#   total suspected deaths in the DRC, and cases (and one death)
#   detected in Uganda — from press and situation reports. There is no line list and no
#   temporal information: no onset dates, no epidemic curve, no
#   per-case data. The model also has no knowledge of the situation
#   on the ground (case definitions, testing capacity, affected
#   areas, interventions, reporting completeness). Every estimate is
#   a model-based extrapolation from sparse summary statistics under
#   strong assumptions rather than a measurement.
# - *LLM-driven reimplementation.* The model code, priors,
#   convolution implementation and analysis were drafted by a
#   language model from the published McCabe et al.
#   [mccabe2026](@cite) report and the
#   companion delay reanalysis, then reviewed and revised. Not
#   independently replicated against the authors' code.
# - *Prior-driven inference where data is scarce.* A dozen suspected
#   exports, ~$10^2$ deaths, and a single reported-case total give
#   little information about $\tau$, $m$, the surveillance dispersion,
#   or the reporting fraction individually. Posteriors track their
#   priors closely.
# - *Inherits McCabe et al.'s epidemiological assumptions and core
#   model structures.* Exponential growth from a single zoonotic
#   seed; the underlying case trajectory is treated as a
#   deterministic function of the latent state (only the
#   observation counts carry sampling noise via Poisson / NegBinomial)
#   rather than a stochastic incidence process; the cumulative-case
#   / deaths convolution structure for Method 2; the geographic-
#   spread / detection-window structure for Method 1; no spatial
#   structure beyond the Ituri / Nord Kivu split; no time series
#   of cases or deaths.
# - *Onset-to-death delay anchored on Isiro 2012.* A single-
#   outbreak fit; the delay distribution reporting here follows
#   [charniga2024](@cite) but cross-outbreak heterogeneity is
#   unmodelled. The baseline fit uses the full Rosello onset-to-death
#   distribution, as in McCabe et al. The
#   [delay sensitivity](#Delay-sensitivity) section refits the joint
#   model with the community-only delay (the $n = 5$ cases who died
#   without admission, weak evidence of a shorter delay) to show how
#   much the outbreak-size estimate leans on the delay assumption.
# - *Detection window is weakly motivated.* $w$ lumps incubation and
#   onset-to-detection together — both poorly characterised for BVD —
#   so the quantity itself is loosely defined. Its prior is even less
#   grounded: it simply spans the 10–20 day windows McCabe et al. sweep,
#   with no independent estimate behind it, so the exports stream leans
#   on an assumption rather than data.
# - *Not all Uganda cases are confirmed exports.* The exports
#   likelihood treats every Uganda case as imported from DRC, but the
#   12 suspected cases reported in Kampala are not all confirmed to be
#   importations — some may reflect onward transmission within Uganda
#   or unrelated suspected cases later discarded. Counting non-exports
#   as exports inflates the export signal and biases the implied
#   outbreak size and ascertainment.
# - *Selection bias in deaths-among-exports.* The deaths-among-
#   exports likelihood assumes Uganda's surveillance retains detected
#   exports through to any subsequent death. If the system loses
#   cases that progress to death, the observed count is biased
#   downward and the constraint it places on $T$ is overstated.
# - *Deaths-among-exports is an approximate construction.* The expected
#   count weights the exported at-risk person-time by the onset-to-death
#   CDF $F_d(T-s)$ rather than convolving the death delay against the
#   exported-case incidence; this treats the cohort present at time $s$
#   as if infected at $s$. A more direct construction would convolve the
#   onset-to-death delay against the exported-case incidence trajectory.
# - *Ascertainment partially pooled, not separately identified.*
#   Uganda's exported-case ascertainment $p_{\text{Uganda}}$ and DRC's
#   reported-case ascertainment $p_{\text{DRC}}$ share a logit-scale
#   hyperprior. With a handful of suspected exports and one export
#   death the Uganda fraction is weakly identified and leans on the
#   pooled mean and the DRC side.
# - *Observation streams share one case pool.* The four streams are
#   modelled as conditionally independent given the latent cumulative
#   incidence, but they observe overlapping individuals — exported
#   cases are a subset of all cases (and may also be DRC-reported),
#   and expected DRC deaths are computed over all incidence including
#   those who travelled. Conditional independence ignores this
#   individual-level overlap, so it can double-count evidence and
#   understate uncertainty. The effect is small here because the
#   Uganda counts are small.
# - *Data conflict not explored in detail.* We combine four data
#   streams jointly but have not systematically checked whether they
#   conflict — whether, say, the exports and the deaths streams imply
#   different outbreak sizes. Characterising data-source properties and
#   conflict is part of the modelling workflow we otherwise follow
#   [abbott_workflow](@cite); a fuller treatment is left for future
#   work.
#
#md # ```@raw html
#md # <details><summary>Load packages and seed the RNG</summary>
#md # ```

using Turing
using Turing: to_submodel, @varname
using Distributions
using StatsFuns: logit, logistic
using DataFrames: DataFrame
import CSV
using Random
using Markdown
using Dates: Date, Day
using BVDOutbreakSize
using BVDOutbreakSize: integrate_cumulative, integrate_exports_deaths,
                       expected_deaths
import CairoMakie

## Render figures at higher resolution so they stay crisp in the docs.
CairoMakie.activate!(type = "png", px_per_unit = 3)

Random.seed!(20260518)

#md # ```@raw html
#md # </details>
#md # ```

# ## Methods
#
# ### Data
#
# The analysis uses a handful of aggregate counts collated from
# situation reports and news coverage: the suspected cases and
# suspected deaths reported in the DRC, the cases (and any deaths)
# detected among travellers to Uganda, and the daily cross-border
# traveller volume and source-area population from the McCabe et al.
# report [mccabe2026](@cite). All are point-in-time totals as of the
# data cut-off, not time series, and the suspected counts are
# unconfirmed. The table below lists each figure with its source. The
# source population is treated as fixed (census data); the daily
# outbound traveller volume is given a normal prior centred at the
# McCabe et al. figure with an SD covering point-of-entry variation.

#md # ```@raw html
#md # <details><summary>Loading observations and building the data table</summary>
#md # ```

obs = load_observations()
observations_table = DataFrame(
    field = [
        "exported_cases",
        "exports_deaths",
        "total_deaths",
        "reported_cases",
        "daily_outbound_travellers",
        "daily_outbound_travellers_sd",
        "source_population",
    ],
    value = [
        obs.exported_cases,
        obs.exports_deaths,
        obs.total_deaths,
        obs.reported_cases,
        obs.daily_outbound_travellers,
        obs.daily_outbound_travellers_sd,
        obs.source_population,
    ],
    source = [
        obs.sources.exported_cases,
        obs.sources.exports_deaths,
        obs.sources.total_deaths,
        obs.sources.reported_cases,
        obs.sources.daily_outbound_travellers,
        obs.sources.daily_outbound_travellers_sd,
        obs.sources.source_population,
    ],
);

const ITURI_POPULATION    = obs.source_population
const ITURI_DAILY_TRAVEL  = obs.daily_outbound_travellers
const EXPORTED_CASES      = obs.exported_cases
const EXPORTS_DEATHS      = obs.exports_deaths

#md # ```@raw html
#md # </details>
#md # ```

observations_table #hide

# ### Model
#
# #### Model overview
#
# We model a single outbreak seeded by one zoonotic introduction that
# then grows exponentially, so the cumulative number of people ever
# infected by outbreak age $s$ is $C(s) = \exp(r s)$, set by a growth
# rate $r$ (equivalently a doubling time). We never observe infections
# directly. Instead each available data stream observes a thinned,
# delayed or transformed view of that same latent incidence curve.
# Reported cases in the DRC are an ascertained fraction of the
# cumulative cases. Suspected deaths in the DRC are the case fatality
# ratio applied to past incidence, convolved with the onset-to-death
# delay. Cases exported to Uganda are the fraction of recent cases that
# crossed the border, set by the travel rate and a detection window.
# Deaths among the exported cases are those cases weighted by their
# probability of having died by now.
#
# Fitting all four streams together gives the posterior for the latent
# cumulative case count $C(T)$ at the report date $T$ — the quantity we
# care about — while sharing the growth, delay, fatality and
# ascertainment parameters across the streams that depend on them.
#
# In implementation terms, the model is assembled from small reusable
# Turing [ge2018turing](@cite) submodels. Each
# *building-block submodel* owns the maths and priors for one epidemic
# parameter family. The *observation submodels* assemble those blocks,
# introduce the forward integrals and the likelihoods, and tie one data
# stream to the latent state. The *composers* combine the observation
# submodels into the per-stream fits and the joint fit. The diagram
# below traces that flow:
#
# {{MODEL_DIAGRAM}}
#
# Reading top to bottom:
#
# 1. **Building-block submodels** — one per parameter family
#    (growth, onset-to-death delay, CFR, detection window, daily
#    traveller volume, surveillance dispersion, ascertainment). Each
#    samples its own
#    priors and returns a small NamedTuple of values. These sections
#    introduce only the maths for their own parameters.
# 2. **Observation submodels** — exports, deaths, cases, deaths-among-
#    exports. Each takes the growth state as input, introduces the
#    forward integral it needs and the likelihood, and ties one data
#    stream to the latent $C(T)$.
# 3. **Composers** — one per analysis: the four single-stream fits, a
#    two-stream reimplementation of the report's methods (exports and
#    deaths), and the full joint fit. Each samples the building blocks
#    and the relevant observation
#    submodels. A composer conditionally includes only the likelihoods
#    for the streams it uses, so a single-stream fit never instantiates
#    the other observation submodels.
# 4. **Inference** — prior predictive, the four No-U-Turn Sampler
#    (NUTS) fits, posterior
#    summaries, posterior-predictive plots.
# 5. **Counterfactual, forecast and sense check** — a
#    no-onward-transmission lower bound on cumulative deaths, a
#    one-week-ahead forecast, and a `Turing.fix`-pinned reproduction
#    of Method 2 main scenario via the exports-and-deaths
#    composer.

# #### Building-block submodels
#
# Each building-block submodel introduces only the mathematical objects
# and priors for one parameter family; the likelihoods and forward
# integrals enter later, in the observation submodels that use them.
#
# The implementation approach taken here is based on the hantavirus
# modelling project [hantavirus_2026](@cite): Mooncake
# [mooncake_jl](@cite) automatic differentiation, the
# Integrals [integrals_jl](@cite) quadrature helpers, a NaN-safe
# NegBinomial, FlexiChains, and PairPlots [pairplots_jl](@cite) with
# AlgebraOfGraphics [danisch2021makie](@cite) for the figures.

# ##### Growth — exponential
#
# The outbreak is seeded $T$ days ago by a single zoonotic case and
# grows exponentially with doubling time $\tau$, giving the cumulative-
# incidence trajectory
#
# ```math
# C(s) = \exp(r\,s), \qquad r = \frac{\log 2}{\tau}, \tag{1}
# ```
#
# so that the cumulative case count at the cut-off is $C(T) = 2^m$
# with $m = T/\tau$ the number of doublings since seeding. McCabe et al.
# vary the doubling time over a sensitivity sweep of 7 / 14 / 21
# days; here $\tau$ has a LogNormal prior centred at the main scenario
# (14 d) with log-SD 0.4, giving a 95% prior interval of roughly
# $(6, 31)$ d that encompasses the full sweep:
#
# ```math
# \tau \sim \mathrm{LogNormal}(\log 14,\ 0.4). \tag{2}
# ```
#
# Rather than sampling $\tau$ and $T$ directly (which are ridge-
# correlated through $C(T) = \exp(r T)$), the model samples $\tau$ and
# the *doubling-time multiplier* $m = T/\tau$. Then $C(T) = 2^m$ is
# near-orthogonal to $\tau$. $m$ is centred at 7 ($C(T) = 2^7 = 128$)
# with SD 2.5, truncated to $(0, 13]$:
#
# ```math
# m \sim \mathrm{Normal}(7,\ 2.5)\ \text{on}\ (0, 13]. \tag{3}
# ```
#
# This gives 95% prior support of roughly $m \in (2, 12)$, i.e.
# $C(T) \in (4, 4000)$. The range is chosen to span the number of
# doublings plausible under the doubling times McCabe et al. sweep
# (7–21 days) over a likely few weeks to months of spread since
# seeding — it is motivated by their scenario *settings*, not by their
# reported outbreak sizes. The hard upper bound at 13 caps $C(T)$ at
# ~8000. The growth rate $r$ and the elapsed time $T = m\cdot\tau$ are
# exposed as deterministics so they appear in posterior tables and
# pair plots.

#md # ```@raw html
#md # <details><summary>Submodel: exponential_growth_model</summary>
#md # ```

@model function exponential_growth_model(;
        tau_prior = LogNormal(log(14), 0.4),
        m_prior   = truncated(Normal(7.0, 2.5);
                              lower = 0, upper = 13.0))
    τ ~ tau_prior
    m ~ m_prior
    r   := log(2) / τ
    T   := m * τ
    C_T := 2.0 ^ m
    cumulative = s -> exp(r * s)
    return (; τ, r, m, T, C_T, cumulative)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Genetic seeding bound
#
# The first eight sequenced cases give a maximum-likelihood phylogeny
# [virological2026](@cite) whose root-to-tip genetic distance, divided
# by the $1.2\times10^{-3}$ substitutions/site/year molecular clock rate
# estimated for the 2013–2016 West African Ebola epidemic
# [holmes2016](@cite), places the time to the most recent common
# ancestor (TMRCA), the age of the oldest internal node of the tree,
# roughly 80 days before the reference date on which the estimate was
# reported (combination per N. Ferguson [ferguson2026](@cite)).
# This is a lower bound on the seeding time $T$, since the TMRCA only
# moves older as more sequences are added.
# We do not know exactly where the floor sits, so we treat it as an
# uncertain threshold $B \sim \mathrm{Normal}(g, \sigma)$ and require
# $T \ge B$, leaving $T$ free above it.
# Marginalising over $B$ gives a soft one-sided likelihood,
#
# ```math
# p_\text{gen}(T) = \Pr[B \le T] = \Phi\!\left(\frac{T - g}{\sigma}\right),
# \qquad g \approx 80\ \text{d}, \ \sigma = 20\ \text{d}. \tag{3a}
# ```

#md # ```@raw html
#md # <details><summary>Submodel: genetic_seeding_model</summary>
#md # ```

@model function genetic_seeding_model(T, tmrca_days::Real;
        tmrca_days_sd::Real)
    ## The molecular-clock TMRCA is a right-censored, noisy reading of the
    ## true seeding time T: deeper or wider sampling only pushes it older,
    ## so we learn the reading is at least `tmrca_days`, i.e. P(read ≥ g).
    tmrca_days ~ censored(Normal(T, tmrca_days_sd); upper = tmrca_days)
    return (; tmrca_days, tmrca_days_sd)
end

# Observing $g$ (`tmrca_days`) at the upper censoring point of
# $\mathrm{Normal}(T, \sigma)$ contributes the log probability of the
# censored upper tail, which is the soft bound of Eq. (3a):
#
# ```math
# \Pr[\mathrm{Normal}(T, \sigma) \ge g]
#   = 1 - \Phi\!\left(\frac{g - T}{\sigma}\right)
#   = \Phi\!\left(\frac{T - g}{\sigma}\right).
# ```

#md # ```@raw html
#md # </details>
#md # ```

# ##### Onset-to-death delay
#
# Following McCabe et al., we assume the symptom-onset-to-death delay is
# gamma distributed with shape $\alpha$ and scale $\theta$, with density
# $f$ and CDF $F_d$:
#
# ```math
# \text{delay} \sim \mathrm{Gamma}(\alpha,\ \theta). \tag{4}
# ```
#
# The McCabe et al. report uses the point estimate of
# [rosello2015](@cite). We instead use the companion Bayesian reanalysis
# of the same Isiro line list [bdbv_linelist_analysis_2026](@cite),
# which re-estimates the delay with uncertainty. We carry that
# uncertainty into the fit through truncated Normal priors centred on
# its estimates:
#
# ```math
# \alpha \sim \mathrm{Normal}^{+}(4.3,\ 1.22), \qquad
# \theta \sim \mathrm{Normal}^{+}(2.6,\ 0.82). \tag{5}
# ```
#
# The delay estimation in that reanalysis follows the recommendations
# of [charniga2024](@cite).

#md # ```@raw html
#md # <details><summary>Submodel: delay_model</summary>
#md # ```

@model function delay_model(;
        alpha_prior = truncated(Normal(4.3, 1.22); lower = 0),
        theta_prior = truncated(Normal(2.6, 0.82); lower = 0))
    α ~ alpha_prior
    θ ~ theta_prior
    return (; α, θ, dist = Gamma(α, θ))
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Case-fatality ratio
#
# The US Centers for Disease Control and Prevention (CDC) summary for
# the two previous BVD outbreaks is $55$ deaths in $169$ cases
# ($\approx 33\%$;
# [CDC outbreak history](https://www.cdc.gov/ebola/outbreaks/index.html)),
# with confidence bands spanning
# roughly $26$-$40\%$. The companion Bundibugyo virus (BDBV) reanalysis
# reports a baseline of $0.47$ ($95\%$ CrI $0.31$-$0.65$) for
# non-healthcare-worker (non-HCW) confirmed cases. The prior on the
# case-fatality ratio is
#
# ```math
# \mathrm{CFR} \sim \mathrm{Beta}(6.6,\ 13.4), \tag{6}
# ```
#
# with mean $0.33$ and $95\%$ interval roughly $0.15$-$0.54$. The mean
# matches the CDC $55/169 \approx 33\%$ figure and the corrected central
# CFR in the 20 May report [mccabe2026update](@cite).

#md # ```@raw html
#md # <details><summary>Submodel: cfr_model</summary>
#md # ```

@model function cfr_model(; cfr_prior = Beta(6.6, 13.4))
    CFR ~ cfr_prior
    return (; CFR)
end

#md # ```@raw html
#md # </details>
#md # ```

# The prior density, with the CDC $0.33$ figure marked, as a sense
# check.

cfr_prior_fig = plot_cfr_prior(Beta(6.6, 13.4)); #hide
cfr_prior_fig #hide

# ##### Detection window
#
# $w$ is the mean time during which a case is still infectious and
# detectable abroad (incubation + onset-to-detection). The prior is
# based on the detection windows McCabe et al. sweep in their Method 1
# scenarios (10, 15 and 20 days): it is centred on their central 15-day
# value with an SD wide enough to cover the 10–20 day range.
#
# ```math
# w \sim \mathrm{Normal}^{+}(15,\ 5)\ \text{days}. \tag{7}
# ```

#md # ```@raw html
#md # <details><summary>Submodel: detection_window_model</summary>
#md # ```

@model function detection_window_model(;
        window_prior = truncated(Normal(15.0, 5.0); lower = 0))
    w ~ window_prior
    return (; w)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Daily traveller volume
#
# The number of people crossing from the source area to Uganda each day
# sets the travel rate in the exports likelihood. We treat it as an
# estimated quantity rather than a fixed input. McCabe et al. Table 3
# records mean weekly passenger counts across seven points of entry; the
# Ituri-side daily total of $1871$ is a sample mean across roughly
# $15$-$21$ point-of-entry-weeks. We use a Normal prior centred on
# $1871$ with SD $200$ ($\approx 10\%$ CV), truncated at zero, covering
# point-of-entry variation and the sitrep sampling uncertainty; the
# source population is kept fixed (census).

#md # ```@raw html
#md # <details><summary>Submodel: traveller volume</summary>
#md # ```

@model function traveller_volume_model(;
        mean::Real = ITURI_DAILY_TRAVEL,
        sd::Real   = ITURI_DAILY_TRAVEL_SD)
    daily_travellers ~ truncated(Normal(mean, sd); lower = 0)
    return (; daily_travellers)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Surveillance dispersion
#
# We assume the passive-surveillance counts are reported with negative
# binomial observation error around their expected value, using the
# same error model for both streams it applies to — suspected deaths
# and reported cases in the DRC — with a single shared dispersion $k$
# because they come from the same surveillance system. Under the
# mean-$\mu$ / dispersion-$k$ parameterisation a count $Y$ has
#
# ```math
# Y \sim \mathrm{NegBinomial}(\mu,\ k), \qquad
# \mathrm{Var}(Y) = \mu + \frac{\mu^2}{k}. \tag{8}
# ```
#
# The dispersion captures passive-surveillance noise (under-reporting
# that varies by district, weekend reporting effects, and batched
# updates), not transmission heterogeneity.
# We judge this noise to be substantial, so a priori we expect
# meaningful overdispersion rather than near-Poisson counts.
# Following the Stan prior-choice recommendations
# [stan_prior_choice](@cite), the dispersion is sampled on the
# $1/\sqrt{k}$ scale, which behaves like a standard deviation, with a
# weakly-informative prior centred on that expected overdispersion,
#
# ```math
# 1/\sqrt{k} \sim \mathrm{Normal}^{+}(0.6,\ 0.2), \tag{9}
# ```
#
# giving $k$ a prior median near $3$ with a 90% range of about $1$-$14$.
# Because the prior allows near-Poisson counts, $k$ itself ranges over
# many orders of magnitude, so the pair plots and summary table show
# dispersion on both the sampled $1/\sqrt{k}$ scale, which is easier to
# read, and the more familiar $k$ scale.
# Because each stream contributes essentially one aggregate count, $k$
# is only weakly identified, so this prior carries the inference and is
# set to reflect the overdispersion we expect from passive surveillance.
# This extends the McCabe et al. report, which uses a Poisson likelihood
# for the Method 2 deaths and does not model the reported case counts at
# all; the negative binomial adds overdispersion to absorb
# passive-surveillance noise.

#md # ```@raw html
#md # <details><summary>Submodel: surveillance_dispersion_model</summary>
#md # ```

@model function surveillance_dispersion_model(;
        inv_sqrt_k_prior = truncated(Normal(0.6, 0.2); lower = 0))
    inv_sqrt_k ~ inv_sqrt_k_prior
    k := 1.0 / (inv_sqrt_k^2 + eps(typeof(inv_sqrt_k)))
    return (; k, inv_sqrt_k)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Ascertainment — partial pooling between DRC and Uganda
#
# Two surveillance systems detect cases: DRC passive surveillance (the
# reported suspected-case count) and Uganda's point-of-entry / hospital
# surveillance (the exported-case count). Each captures only a fraction
# of the true cases passing through it, and each fraction is informed
# by essentially a single aggregate data point — the one reported-case
# total and the one export count — so neither is well identified on its
# own. We therefore centre the prior on an assumed reporting fraction
# of 25% and partially pool the two fractions so they share strength:
# treating them as identical would conflate two different systems,
# while treating them as independent would leave the Uganda fraction
# almost wholly prior-driven.
#
# Both ascertainment fractions $p_{\text{DRC}}$ and $p_{\text{Uganda}}$
# share a logit-scale hyperprior with mean $\mu$ and SD $\tau$:
#
# ```math
# \mu \sim \mathrm{Normal}(\mathrm{logit}(0.25),\ 1),
# \qquad
# \tau \sim \mathrm{Normal}^{+}(0,\ 0.5), \tag{10}
# ```
#
# ```math
# \mathrm{logit}(p_{\text{DRC}}) \sim
#     \mathrm{Normal}(\mu,\ \tau),
# \qquad
# \mathrm{logit}(p_{\text{Uganda}}) \sim
#     \mathrm{Normal}(\mu,\ \tau), \tag{11}
# ```
#
# with $p = \mathrm{logistic}(\mathrm{logit}\,p)$. Here $\tau$ is the
# pooling strength: small $\tau$ pulls the two fractions together (the
# shared-fraction limit), large $\tau$ lets them move independently
# (the separate-fraction limit). The cases likelihood uses
# $p_{\text{DRC}}$; the two Uganda-side likelihoods use
# $p_{\text{Uganda}}$.
#
# We sample this prior in its non-centred form: draw offsets
# $z_{\text{DRC}}, z_{\text{Uganda}} \sim \mathrm{Normal}(0, 1)$ and set
# $\mathrm{logit}(p) = \mu + \tau z$. This is the same prior but avoids
# the funnel geometry of the centred form, which gave hundreds of
# divergent transitions.

#md # ```@raw html
#md # <details><summary>Submodel: pooled_ascertainment_model</summary>
#md # ```

@model function pooled_ascertainment_model(;
        mu_prior  = Normal(logit(0.25), 1.0),
        tau_prior = truncated(Normal(0, 0.5); lower = 1e-4))
    μ_logit  ~ mu_prior
    τ_logit  ~ tau_prior
    z_drc    ~ Normal(0, 1)
    z_uganda ~ Normal(0, 1)
    logit_p_drc    = μ_logit + τ_logit * z_drc
    logit_p_uganda = μ_logit + τ_logit * z_uganda
    p_drc    := logistic(logit_p_drc)
    p_uganda := logistic(logit_p_uganda)
    return (; μ_logit, τ_logit, p_drc, p_uganda)
end

#md # ```@raw html
#md # </details>
#md # ```

# We model both the deaths and the reported cases with a negative
# binomial, so we define one small constructor for it and share it.
# It is parameterised by mean $\mu$ and dispersion $k$ (so the variance
# is given by equation (8)), with NaN / Inf-safe clamping on the
# success probability so extreme NUTS proposals during warmup do not
# trip the distribution domain check. It is used by the deaths and
# cases observation submodels below.

#md # ```@raw html
#md # <details><summary>Function: safe_nbinomial</summary>
#md # ```

function safe_nbinomial(k, μ)
    p_raw = k / (k + max(μ, eps(typeof(μ))))
    p = isfinite(p_raw) ?
        clamp(p_raw, eps(typeof(k)), one(k) - eps(typeof(k))) :
        eps(typeof(k))
    return NegativeBinomial(k, p)
end

#md # ```@raw html
#md # </details>
#md # ```

# #### Observation submodels
#
# With the building blocks in place, each observation submodel takes
# the growth state as input, introduces the forward integral it needs,
# and ties one data stream to the latent $C(T)$. The forward integrals
# (the at-risk person-time integral for exports, the gamma convolution
# for deaths, and the deaths-among-exports convolution) are solved
# numerically. Each submodel introduces its likelihood by
# referring back to the parameters defined in equations (1)-(11).

# ##### Exports — Method 1 (geographic spread)
#
# Each case in the source population travels to Uganda on any given
# day with probability
# $q = \text{daily travellers}/\text{source population}$,
# treating cases as exchangeable with the general population. A case
# is *detection-eligible* for $w$ days from infection (equation (7)).
# For a case infected at outbreak age $s \leq T$, the accumulated
# probability of being detected in Uganda by $T$ is
#
# ```math
# P(\text{detected by } T \mid \text{infected at } s)
#     = q \cdot \min(T - s,\ w). \tag{12}
# ```
#
# Splitting at $s = T - w$ (full window elapsed before $T-w$, partial
# window after) and summing across incidence $i(s)$ gives the full
# export integral
#
# ```math
# \mathbb{E}[\text{exports by }T]
#     = q \cdot \Bigl[ w \cdot C(T-w)
#          + \int_{T-w}^{T} i(s) \, (T - s) \, ds \Bigr], \tag{13}
# ```
#
# which integration by parts collapses to the at-risk person-time form
# using the cumulative-incidence trajectory $C(s)$ of
# equation (1):
#
# ```math
# \mathbb{E}[\text{exports by }T] = q \cdot \int_{T-w}^{T} C(s)\, ds. \tag{14}
# ```
#
# For exponential growth this evaluates to
# $q\cdot(C(T) - C(T-w))/r$. We evaluate equation (14) numerically so
# the same form works for any growth parameterisation, scale by the
# Uganda ascertainment fraction $p_{\text{Uganda}}$ (equation (11)),
# and apply a Poisson likelihood:
#
# ```math
# \mu_e = p_{\text{Uganda}} \cdot q \cdot \int_{T-w}^{T} C(s)\, ds,
# \qquad
# Y_{\text{exports}} \sim \mathrm{Poisson}(\mu_e). \tag{15}
# ```
#
# !!! note "Comparison with McCabe et al. / Imai 2020"
#     McCabe et al. use the small-$rw$ simplification
#     $\mu_e \approx q\cdot w\cdot C(T)$, the limit of equation (14)
#     as $r \to 0$.
#     For BVD's prior range $rw \in 0.33 - 2.0$ the simplification
#     under-estimates $C(T)$ by roughly $15$-$57\%$. We use a Poisson
#     likelihood for the detected exports; at the small detection
#     probability here ($p \approx q\cdot w \approx 6\cdot 10^{-3}$) it
#     is indistinguishable from a binomial detection model.

#md # ```@raw html
#md # <details><summary>Submodel: exports_model</summary>
#md # ```

@model function exports_model(
        exported_cases::Union{Missing, Integer},
        growth_state, p_uganda::Real;
        source_population::Real = ITURI_POPULATION,
        window                  = detection_window_model(),
        traveller               = traveller_volume_model())

    cumulative = growth_state.cumulative
    T          = growth_state.T

    window_state ~ to_submodel(window, false)
    w = window_state.w

    travel_state ~ to_submodel(traveller, false)
    daily_travellers = travel_state.daily_travellers

    q = daily_travellers / source_population
    expected_exports_T := expected_exports(cumulative, p_uganda, q, T, w)

    exported_cases ~ Poisson(expected_exports_T)

    return (; w, daily_travellers, p_uganda,
              expected_exports = expected_exports_T)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Deaths — Method 2 (back-calculation from deaths)
#
# Expected cumulative deaths by $T$ from a single seeding case is the
# CFR-weighted convolution of the cumulative-incidence trajectory
# $C(s)$ (equation (1)) with the onset-to-death density $f$
# (equation (4)):
#
# ```math
# \mathbb{E}[D(T)] = \mathrm{CFR} \cdot
#     \int_0^T e^{r s}\, f(T - s)\, ds. \tag{16}
# ```
#
# For a gamma delay this integral has an exact closed form carrying a
# $\gamma(\alpha, (\beta + r)T)/\Gamma(\alpha)$ factor from the finite
# upper limit; McCabe et
# al. use the large-$T$ simplification
# $D(T) \approx \mathrm{CFR}\cdot C(T)\cdot(1 + r/\beta)^{-\alpha}$
# (valid for $T \gtrsim 12/(\beta+r)$), which
# drops that factor. We evaluate equation (16) numerically instead,
# which is exact. The
# observed deaths follow the NegBinomial likelihood of equation (8)
# with the dispersion $k$ of equation (9), supplied by the composer so
# it can be shared with the cases likelihood:
#
# ```math
# Y_{\text{deaths}} \sim \mathrm{NegBinomial}(\mathbb{E}[D(T)],\ k). \tag{17}
# ```

#md # ```@raw html
#md # <details><summary>Submodel: deaths_model</summary>
#md # ```

@model function deaths_model(
        total_deaths::Union{Missing, Integer},
        growth_state, k::Real;
        delay = delay_model(),
        cfr   = cfr_model())

    C_T = growth_state.C_T
    r   = growth_state.r
    T   = growth_state.T

    delay_state ~ to_submodel(delay, false)
    cfr_state   ~ to_submodel(cfr, false)

    CFR = cfr_state.CFR

    ## NaN-safe clamp: extreme NUTS proposals during warmup can push
    ## the expected count to NaN / Inf.
    raw_deaths         = expected_deaths(CFR, r, T, delay_state.dist)
    expected_deaths_T := isfinite(raw_deaths) ?
        max(raw_deaths, eps(typeof(raw_deaths))) :
        eps(typeof(raw_deaths))

    total_deaths ~ safe_nbinomial(k, expected_deaths_T)

    return (; CFR, delay_dist = delay_state.dist, expected_deaths_T)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Cases — ascertainment extension#
# Methods 1 and 2 use exports and deaths only. Reported
# suspected cases from the same passive-surveillance system carry
# information about $C(T)$ once the DRC ascertainment fraction $p_{\text{DRC}}$
# (equation (11)) is introduced:
#
# ```math
# \mu_c = p_{\text{DRC}} \cdot C(T),
# \qquad
# Y_{\text{cases}} \sim \mathrm{NegBinomial}(\mu_c,\ k). \tag{18}
# ```
#
# The dispersion $k$ (equation (9)) is shared with the deaths
# likelihood; both are sampled once by the composer.

#md # ```@raw html
#md # <details><summary>Submodel: cases_model</summary>
#md # ```

@model function cases_model(
        reported_cases::Union{Missing, Integer},
        growth_state, k::Real, p_drc::Real)

    C_T = growth_state.C_T

    raw_reports        = p_drc * C_T
    expected_reports := isfinite(raw_reports) ?
        max(raw_reports, eps(typeof(raw_reports))) :
        eps(typeof(raw_reports))

    reported_cases ~ safe_nbinomial(k, expected_reports)

    return (; p_drc, expected_reports)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Deaths among exports
#
# Uganda reports a single death among its detected exports. That count
# is informative. If the exports happened long ago, we would expect
# more deaths among them by now under the onset-to-death gamma, so the
# observed deaths-among-exports bound how recently the exports occurred
# and help constrain $T$ (and $C(T)$). The expected count reuses the at-risk
# export integrand of equation (14) but weights each case by its
# probability of having died by $T$, the onset-to-death CDF
# $F_d(T - s)$ (equation (4)), then scales by the CFR, the travel rate
# $q$ and the Uganda ascertainment fraction $p_{\text{Uganda}}$:
#
# ```math
# \mathbb{E}[D_{\text{Uganda}}]
#     = \mathrm{CFR} \cdot p_{\text{Uganda}} \cdot q
#       \cdot \int_{T-w}^{T} C(s)\, F_d(T - s)\, ds. \tag{19}
# ```
#
# Equation (19) is evaluated numerically, writing $F_d$ as the inner
# integral of the density $f$ so the whole expression differentiates
# through $f$ alone (the reverse-mode AD backend does not support the
# gamma CDF shape-parameter derivative). The detection window $w$ and
# daily traveller volume are shared with the exports likelihood so the
# two Uganda-side observations use the same person-time.
#
# Uganda's export deaths are point-of-entry / hospital-detected, so
# their dates are recorded directly and carry information beyond the
# total count: a death seen early bounds how old the outbreak can be. We
# model the detected export deaths as an inhomogeneous Poisson process
# with cumulative intensity $\mathbb{E}[D_{\text{Uganda}}(t)]$
# (equation (19), at any elapsed time $t$) and use its time-resolved
# likelihood. Splitting $[0, T]$ at the earliest
# dated death (offset $\Delta_1$ before the cut-off, elapsed time
# $T-\Delta_1$): before it no export death was seen, contributing one
# continuous survival term over $[0, T-\Delta_1]$; from that day to the
# cut-off each day $d$ carries a Poisson count of the export deaths that
# day, with bin mean $\mu_d$,
#
# ```math
# \log L = \sum_d y_d \log \mu_d - \mathbb{E}[D_{\text{Uganda}}(T)],
# \quad
# \mu_d = \mathbb{E}[D_{\text{Uganda}}(t_d)]
#         - \mathbb{E}[D_{\text{Uganda}}(t_{d-1})]. \tag{20}
# ```
#
# The continuous term collapses the long pre-death zero stretch into a
# single weight, so there is no per-day vector of zeros, while the
# recent window is resolved per day. The number of daily bins is fixed
# by the earliest death's date, so the likelihood is well defined even
# though $T$ is latent; with one death the series is a single count, and
# it takes more dated deaths as they are reported.
#
# !!! note "Selection-bias caveat"
#     This assumes Uganda's surveillance retains detected exports
#     through to any subsequent death. If the system instead loses
#     cases that progress to death, the observed deaths-among-exports
#     count is selection-biased downward and the constraint it places
#     on $T$ is overstated.

#md # ```@raw html
#md # <details><summary>Submodel: exports_deaths_model</summary>
#md # ```

@model function exports_deaths_model(
        export_deaths_daily::AbstractVector,
        growth_state, CFR::Real, delay_dist, p_uganda::Real;
        pre_start_deaths::Union{Missing, Integer} = 0,
        window::Real,
        daily_travellers::Real,
        source_population::Real = ITURI_POPULATION)

    cumulative = growth_state.cumulative
    T          = growth_state.T
    q          = daily_travellers / source_population
    n          = length(export_deaths_daily)   # days from earliest death to cut-off

    Λ(t) = expected_exports_deaths(
        cumulative, delay_dist, CFR, p_uganda, q, t, window)

    ## Pre-death zero stretch as one Poisson observed at 0; `missing`
    ## generates it for predictive checks (see equation (20)).
    pre = T - n > zero(T) ? Λ(T - n) : zero(T)
    pre_start_deaths ~ Poisson(max(pre, zero(pre)))

    ## Carry the upper edge forward so each Λ is evaluated once.
    λlo = pre
    for i in 1:n
        λhi = Λ(T - n + i)
        μ_day = max(λhi - λlo, eps(typeof(λhi)))
        export_deaths_daily[i] ~ Poisson(μ_day)
        λlo = λhi
    end

    return (;)
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### First export detection — timing survival term
#
# The same logic applies to the *first detected export case* (Uganda's
# first hospital admission), using the at-risk export person-time
# intensity $\mathbb{E}[\text{exports}(t)]$ (equation (15)) in place of
# the export-death intensity. With $\Delta$ the offset from the first
# admission date to the cut-off and $t_1 = T - \Delta$,
#
# ```math
# \Pr(\text{no export detected before } t_1)
#     = \exp\!\bigl(-\mathbb{E}[\text{exports}(t_1)]\bigr). \tag{22}
# ```
#
# As with the export-death term, this is one-sided and only marginally
# constrains the posterior because the Uganda detections sit only days
# before the cut-off.
# Passing `delta = missing` makes the submodel a no-op.

#md # ```@raw html
#md # <details><summary>Submodel: exports_detection_timing_model</summary>
#md # ```

@model function exports_detection_timing_model(
        growth_state, p_uganda::Real;
        delta::Union{Missing, Real},
        pre_detection_exports::Union{Missing, Integer} = 0,
        window::Real,
        daily_travellers::Real,
        source_population::Real = ITURI_POPULATION)

    if !ismissing(delta)
        cumulative = growth_state.cumulative
        T          = growth_state.T
        t1         = T - delta
        q          = daily_travellers / source_population
        survived_exports := t1 <= zero(T) ? zero(T) :
            expected_exports(cumulative, p_uganda, q, t1, window)
        ## No detection before t1 as a Poisson observed at 0; `missing`
        ## generates it for predictive checks (see equation (22)).
        pre_detection_exports ~ Poisson(max(survived_exports, zero(T)))
    end

    return (;)
end

#md # ```@raw html
#md # </details>
#md # ```

# #### Composers
#
# These composers combine the building blocks into the full model for
# each analysis. McCabe et al. invert a
# deterministic summary formula at fixed nuisance parameters; here we
# sample all of them — growth, delay, CFR, detection
# window, traveller volume, dispersion, ascertainment — and condition
# on the observed counts. Each composer conditionally includes only the
# likelihoods
# for the streams it carries, so a single-stream composer never
# instantiates the other observation submodels (and so a discrete
# stream is never left sampled).
# Of the observation streams, the geographic-spread exports reproduce
# McCabe et al.'s Method 1 and the back-calculation from deaths their
# Method 2; the reported-cases ascertainment, the deaths-among-exports,
# the export-detection-timing, and the genetic seeding terms are
# extensions with no counterpart in McCabe et al.
#
# The joint composer samples a single dispersion scale $k$ and passes it
# to both the deaths and cases likelihoods, so they share one passive-
# surveillance noise scale. It also samples a single pooled set of
# ascertainment fractions, threading $p_{\text{DRC}}$ to the cases
# likelihood and $p_{\text{Uganda}}$ to the two Uganda-side likelihoods. The
# window $w$ and daily traveller volume sampled by the exports
# likelihood are reused by the deaths-among-exports likelihood so the
# two share person-time.

# ##### Exports-only fit — Method 1 analogue

#md # ```@raw html
#md # <details><summary>Composer: exports-only fit</summary>
#md # ```

@model function exports_only_model(
        exported_cases::Union{Missing, Integer};
        growth        = exponential_growth_model(),
        exports       = exports_model,
        ascertainment = pooled_ascertainment_model())

    growth_state ~ to_submodel(growth, false)
    asc_state    ~ to_submodel(ascertainment, false)

    exports_state ~ to_submodel(
        exports(exported_cases, growth_state, asc_state.p_uganda), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Deaths-only fit — Method 2 analogue

#md # ```@raw html
#md # <details><summary>Composer: deaths-only fit</summary>
#md # ```

@model function deaths_only_model(
        total_deaths::Union{Missing, Integer};
        growth     = exponential_growth_model(),
        deaths     = deaths_model,
        dispersion = surveillance_dispersion_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    k = dispersion_state.k

    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Cases-only fit — ascertainment extension
#md # ```@raw html
#md # <details><summary>Composer: cases-only fit</summary>
#md # ```

@model function cases_only_model(
        reported_cases::Union{Missing, Integer};
        growth        = exponential_growth_model(),
        cases         = cases_model,
        dispersion    = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k = dispersion_state.k

    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k, asc_state.p_drc), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Deaths-among-exports-only fit
#md # ```@raw html
#md # <details><summary>Composer: exports-deaths-only fit</summary>
#md # ```

@model function exports_deaths_only_model(
        export_deaths_daily::AbstractVector;
        growth        = exponential_growth_model(),
        delay         = delay_model(),
        cfr           = cfr_model(),
        window        = detection_window_model(),
        traveller     = traveller_volume_model(),
        exports_deaths_model = exports_deaths_model,
        ascertainment = pooled_ascertainment_model(),
        source_population::Real = ITURI_POPULATION,
        pre_start_deaths::Union{Missing, Integer} = 0)

    growth_state ~ to_submodel(growth, false)
    delay_state  ~ to_submodel(delay, false)
    cfr_state    ~ to_submodel(cfr, false)
    window_state ~ to_submodel(window, false)
    asc_state    ~ to_submodel(ascertainment, false)

    travel_state ~ to_submodel(traveller, false)
    daily_travellers = travel_state.daily_travellers

    exports_deaths_state ~ to_submodel(
        exports_deaths_model(export_deaths_daily, growth_state,
            cfr_state.CFR, delay_state.dist, asc_state.p_uganda;
            pre_start_deaths = pre_start_deaths,
            window           = window_state.w,
            daily_travellers = daily_travellers,
            source_population = source_population),
        false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### Joint fit — full posterior over $C(T)$ from all data streams
#
# The joint composer is the same generative process conditioned on all four
# observed streams simultaneously. Each stream argument may be passed
# as `missing` to drop it; the matching likelihood is then not
# instantiated, so the composer doubles as a generator (all streams
# missing) for the prior- and posterior-predictive checks.

#md # ```@raw html
#md # <details><summary>Composer: joint fit</summary>
#md # ```

@model function bvd_joint(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer},
        reported_cases::Union{Missing, Integer} = missing,
        export_deaths_daily::AbstractVector = Int[];
        growth        = exponential_growth_model(),
        exports       = exports_model,
        deaths        = deaths_model,
        cases         = cases_model,
        exports_deaths_model = exports_deaths_model,
        exports_detection_timing = exports_detection_timing_model,
        dispersion    = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model(),
        genetic       = nothing,
        source_population::Real = ITURI_POPULATION,
        pre_start_deaths::Union{Missing, Integer} = 0,
        pre_detection_exports::Union{Missing, Integer} = 0,
        first_export_detection_delta::Union{Missing, Real} = missing)

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
        exports(exported_cases, growth_state, p_uganda), false)
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)
    cases_state ~ to_submodel(
        cases(reported_cases, growth_state, k, p_drc), false)
    exports_deaths_state ~ to_submodel(
        exports_deaths_model(export_deaths_daily, growth_state,
            deaths_state.CFR, deaths_state.delay_dist, p_uganda;
            pre_start_deaths = pre_start_deaths,
            window           = exports_state.w,
            daily_travellers = exports_state.daily_travellers,
            source_population = source_population),
        false)
    detection_timing_state ~ to_submodel(
        exports_detection_timing(growth_state, p_uganda;
            delta            = first_export_detection_delta,
            pre_detection_exports = pre_detection_exports,
            window           = exports_state.w,
            daily_travellers = exports_state.daily_travellers,
            source_population = source_population),
        false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ##### McCabe et al. reimplementation — exports and deaths only
#
# McCabe et al.'s joint configuration uses exactly two data sources: the
# geographic-spread exports (Method 1) and the back-calculation from
# deaths (Method 2). It has no reported-cases ascertainment model and
# no deaths-among-exports likelihood. This composer wraps just
# those two observation submodels so the sense-check can fix the model
# down to exactly the McCabe et al. joint configuration. Either stream may
# be `missing`: passing `missing` for exports recovers a pure Method 2
# (deaths-only) fit without instantiating the exports likelihood.

#md # ```@raw html
#md # <details><summary>Composer: report reimplementation</summary>
#md # ```

@model function imperial_only_model(
        exported_cases::Union{Missing, Integer},
        total_deaths::Union{Missing, Integer};
        growth        = exponential_growth_model(),
        exports       = exports_model,
        deaths        = deaths_model,
        dispersion    = surveillance_dispersion_model(),
        ascertainment = pooled_ascertainment_model())

    growth_state     ~ to_submodel(growth, false)
    dispersion_state ~ to_submodel(dispersion, false)
    asc_state        ~ to_submodel(ascertainment, false)
    k        = dispersion_state.k
    p_uganda = asc_state.p_uganda

    if !ismissing(exported_cases)
        exports_state ~ to_submodel(
            exports(exported_cases, growth_state, p_uganda), false)
    end
    deaths_state ~ to_submodel(
        deaths(total_deaths, growth_state, k), false)

    cumulative_cases := growth_state.C_T
end

#md # ```@raw html
#md # </details>
#md # ```

# ### Model fitting and evaluation
#
# #### Prior predictive check
#
# Before any observation is taken into account, what does the joint
# prior imply about replicated exports, deaths, reported cases and
# deaths among exports? Draws from the prior over the unobserved data
# should bracket the observed counts.

#md # ```@raw html
#md # <details><summary>Sample the joint prior</summary>
#md # ```

prior_chn = sample(bvd_joint(missing, missing, missing, Int[]),
                   Prior(), 2_000; progress = false);

prior_C_table = summary_table(prior_chn, [:cumulative_cases]; digits = 0);

#md # ```@raw html
#md # </details>
#md # ```

prior_C_table #hide

# Pair plot of the prior over the latent quantities — useful for
# spotting prior correlations before any data has been seen.

#md # ```@raw html
#md # <details><summary>Prior pair plot</summary>
#md # ```

prior_pair_fig = plot_pair(prior_chn,
    [:τ, :m, :cumulative_cases, :CFR, :w, :inv_sqrt_k, :k,
     :p_drc, :p_uganda, :τ_logit]);

#md # ```@raw html
#md # </details>
#md # ```

prior_pair_fig #hide

# #### Fitting the models
#
# NUTS [hoffman2014nuts](@cite) with Mooncake [mooncake_jl](@cite)
# reverse-mode automatic differentiation, four chains, 1000 post-warmup
# draws each, with a target acceptance probability of 0.95. Chains
# initialise from the prior
# to keep the sampler away from the boundary of $r$ and $m$. We fit
# the joint model and the four single-stream models so the per-stream
# posteriors over $C(T)$ can be compared with the joint.

#md # ```@raw html
#md # <details><summary>Run the joint and per-stream NUTS fits</summary>
#md # ```

genetic_seeding = T -> genetic_seeding_model(T, obs.genetic_tmrca_days;
    tmrca_days_sd = obs.genetic_tmrca_days_sd)

chn_joint = nuts_sample(
    bvd_joint(obs.exported_cases, obs.total_deaths,
              obs.reported_cases, obs.export_deaths_daily;
              first_export_detection_delta = obs.first_export_detection_delta,
              genetic = genetic_seeding));

chn_exports = nuts_sample(exports_only_model(obs.exported_cases));
chn_deaths  = nuts_sample(deaths_only_model(obs.total_deaths));
chn_cases   = nuts_sample(cases_only_model(obs.reported_cases));
chn_exports_deaths = nuts_sample(
    exports_deaths_only_model(obs.export_deaths_daily));

posterior_C_joint   = vec(Array(chn_joint[:cumulative_cases]));
posterior_C_exports = vec(Array(chn_exports[:cumulative_cases]));
posterior_C_deaths  = vec(Array(chn_deaths[:cumulative_cases]));
posterior_C_cases   = vec(Array(chn_cases[:cumulative_cases]));
posterior_C_exports_deaths =
    vec(Array(chn_exports_deaths[:cumulative_cases]));

#md # ```@raw html
#md # </details>
#md # ```

# #### Fit diagnostics
#
# Fit-quality diagnostics for the joint and per-stream fits: the worst
# R-hat, the smallest bulk effective sample size, and the number of
# divergent transitions.

#md # ```@raw html
#md # <details><summary>Fit diagnostics</summary>
#md # ```

diagnostics_table( #hide
    "joint" => chn_joint, #hide
    "exports (cases)" => chn_exports, #hide
    "exports (deaths)" => chn_exports_deaths, #hide
    "deaths (DRC)" => chn_deaths, #hide
    "cases (DRC)" => chn_cases) #hide

#md # ```@raw html
#md # </details>
#md # ```

# ### Additional analyses
#
# On top of the main analysis we run four supporting analyses.
#
# #### Counterfactual: no-onward-transmission lower bound
#
# Suppose every onward transmission stopped at the report date — the
# estimation / report-generation time, which we denote $T$. The cohort
# already infected by $T$ still carries future expected deaths in the
# onset-to-death tail: a case infected at outbreak age $s$ has died by
# the report date with probability $F_d(T - s)$ (equation (4)), so a
# fraction
# $1 - F_d(T - s)$ of its CFR-weighted contribution has not yet been
# observed. Integrating against the incidence
# $i(s) = r\cdot\exp(r\cdot s)$ from
# equation (1) gives the additional future expected deaths
#
# ```math
# \Delta D = \mathrm{CFR} \cdot \int_0^T r\,\exp(r\,s)
#            \,\bigl(1 - F_d(T - s)\bigr)\,ds, \tag{23}
# ```
#
# and a lower bound on the cumulative-death endpoint of
# $D(T) + \Delta D$, evaluated per posterior draw.
#
# #### One-week-ahead forecast
#
# If the fitted model is taken at face value, what should the next
# week's situation reports show? We continue the fitted exponential
# growth seven days past the report date $T$ and apply the same
# observation models to forecast the cumulative reported cases (DRC),
# deaths (DRC) and exports (Uganda) by $T + 7$, and the *new* counts
# expected over the coming week (cumulative at $T + 7$ minus the count
# already observed). These are posterior-predictive: each draw yields a
# replicated integer count, so the intervals carry both parameter and
# observation uncertainty.
#
# This assumes growth continues unchanged for the week — no
# interventions and no saturation — so it is a "no-change" projection.
#
# #### Delay sensitivity
#
# The deaths back-calculation (equation (16)) depends on the onset-to-
# death delay. The baseline fit anchors the gamma shape $\alpha$ and
# scale
# $\theta$ on the all-deaths Isiro mixture (equation (5)). The companion
# reanalysis [bdbv_linelist_analysis_2026](@cite) also reports a
# *community-only* pathway — the
# $n = 5$ cases who died without hospital admission — with a shorter
# but far more uncertain delay: a shape of about $5.6$
# ($95\%$ CrI $1.0$-$25.9$) and a scale of about $1.4$
# ($95\%$ CrI $0.3$-$9.5$). A shorter delay means deaths appear sooner
# after infection, so a given death count back-calculates to a smaller
# $C(T)$.
#
# We refit the joint model once with the onset-to-death delay priors
# re-anchored on the community-only pathway, building truncated-Normal
# priors from those credible intervals exactly as the baseline delay
# priors (equation (5)) are constructed:
#
# ```math
# \alpha \sim \mathrm{Normal}^{+}(5.6,\ 6.35), \qquad
# \theta \sim \mathrm{Normal}^{+}(1.4,\ 2.35). \tag{24}
# ```
#
# The comparison shows how sensitive the
# outbreak-size estimate is to the delay assumption.
#
# !!! warning "Sensitivity only, not a preferred estimate"
#     The community-only delay is fitted from $n = 5$ deaths, so the
#     evidence is weak and the priors in equation (24) are very wide.
#     This section is included to probe sensitivity, not as a preferred
#     alternative to the baseline.
#
# #### Report reproduction and validation
#
# How does our joint posterior sit against what McCabe et al. reported,
# and how much of any difference is the method rather than the newer
# data? The report itself was revised: the 18 May version
# [mccabe2026](@cite) used $88$ deaths and a central CFR of $30\%$,
# while the 20 May version [mccabe2026update](@cite) used $131$ deaths
# and corrected the central CFR to $33\%$. We work through both in
# sequence so the effect of the report's own deaths-plus-CFR correction
# is visible separately from our newer data and joint method. For each
# version we fit our full joint model to that version's data snapshot
# (`data/report-snapshot.toml` for 18 May, `report-snapshot-20may.toml`
# for 20 May), so the only thing that changes between a version's fit
# and our headline fit is the data.
#
# McCabe et al. Method 2 reports Poisson intervals (no overdispersion,
# $k \to \infty$). We reproduce it by fixing the exports-and-deaths
# composer to their Method 2 central assumptions and dropping exports.
# $1/\sqrt{k}$ is fixed to a small positive value ($k \approx 10^6$,
# Poisson-like) because exactly $0$ gives $k \approx 4.5\times10^{15}$,
# where the NegBinomial saturates and reverse-mode AD returns NaN
# gradients.
#
# As a sense check we ask whether our machinery recovers McCabe et
# al.'s Method 2 headline when given their inputs. The 18 May reported
# Method 2 central estimate is $501$ cases and the 20 May one is $678$
# cases. Each reproduction drops exports so only the deaths likelihood
# is instantiated, conditions on that version's deaths ($88$ for 18 May,
# $131$ for 20 May), and `Turing.fix`-pins the Method 2 main-scenario
# values ($\tau = 14$ d, $\mathrm{CFR} = 30\%$ for 18 May and $33\%$ for
# 20 May, $\alpha = 4.42$, $\beta = 0.388$/d), with the deaths
# NegBinomial made Poisson-like. The only sampled latent is $m$, the
# number of doublings since seeding ($C(T) = 2^m$). A close match
# confirms our back-calculation matches the report; any remaining gap
# to our headline estimate is method (joint fit, exact convolution,
# sampled nuisance parameters) and newer data.
#
# This sense check covers the deaths (Method 2) side. The exports
# (Method 1) side differs by construction: we use the exact cumulative
# integral $q\int_{T-w}^{T} C(s)\,ds$ rather than the small-$rw$
# simplification $q\,w\,C(T)$, so our exports-implied size is expected
# to sit above a Method 1 reproduction (by the $15$-$57\%$ noted
# earlier) rather than match it.

# ## Results
#
# ### Summary
#
# For the response the question that matters is how many people have
# already been infected: the reported counts capture only part of the
# outbreak, and planning for beds, contacts and vaccine needs depends
# on the true total. The numbers below are our current best estimate of
# that total, computed from the joint posterior and refreshed on every
# build. We give 90% credible-interval ranges here; the full 30/60/90%
# intervals are in the tables below.

#md # ```@raw html
#md # <details><summary>Compute the headline ranges</summary>
#md # ```

summary_ranges = let
    C    = posterior_C_joint
    c_lo = round(Int, quantile(C, 0.05))
    c_hi = round(Int, quantile(C, 0.95))
    Tdraws = vec(Array(chn_joint[:T]))
    t_lo = round(Int, quantile(Tdraws, 0.05))
    t_hi = round(Int, quantile(Tdraws, 0.95))
    start_earliest = Date(obs.as_of_date) - Day(t_hi)
    start_latest   = Date(obs.as_of_date) - Day(t_lo)
    f_lo = round(c_lo / obs.reported_cases; digits = 1)
    f_hi = round(c_hi / obs.reported_cases; digits = 1)
    Markdown.parse("""
    - **Current cumulative case load:** a 90% credible interval of
      $(c_lo)–$(c_hi) cases, combining all four data streams (both
      reported and as-yet-unreported).
    - That is roughly $(f_lo)–$(f_hi)× the $(obs.reported_cases) cases
      reported to date, so most infections are not yet reported.
    - **Time since seeding:** a 90% interval of $(t_lo)–$(t_hi) days,
      placing the start of sustained transmission between
      $(start_earliest) and $(start_latest).
    """)
end;

#md # ```@raw html
#md # </details>
#md # ```

summary_ranges #hide

# ### Joint model estimates
#
# Our main result is an estimate of the current cumulative case load —
# both reported and unreported cases — at the report date. It is the
# joint posterior over the cumulative case count, obtained by fitting
# all four data streams together: the cases exported to Uganda, the
# suspected deaths in the DRC, the reported cases in the DRC (with an
# ascertainment component) and the deaths among exported cases in
# Uganda.
#
# We report the cumulative case count first as a credible-interval
# table and then as a posterior density.

#md # ```@raw html
#md # <details><summary>Cumulative case count summary table</summary>
#md # ```

cumulative_cases_summary = summary_table(
    chn_joint, [:cumulative_cases]; digits = 0);

#md # ```@raw html
#md # </details>
#md # ```

cumulative_cases_summary #hide

#md # ```@raw html
#md # <details><summary>Cumulative case count density</summary>
#md # ```

joint_density_fig = plot_cumulative_cases(
    "joint" => posterior_C_joint; scenarios = []);

#md # ```@raw html
#md # </details>
#md # ```

joint_density_fig #hide

# The cumulative case count $C(T) = \exp(r T)$ is set jointly by the
# doubling time $\tau$ (equivalently the growth rate
# $r = \log 2/\tau$) and
# the time since seeding $T$. Read as a calendar date, $T$ places the
# start of sustained transmission at the report date minus $T$ days.
# The left panel below shows the posterior for that start date; the
# right panel shows the joint $(\tau, T)$ posterior, which is
# positively
# correlated: slower growth (larger $\tau$) needs a longer elapsed $T$
# to reach the same observed counts.

#md # ```@raw html
#md # <details><summary>Outbreak start date and (τ, T) posterior</summary>
#md # ```

start_date_fig = plot_start_date_pair(chn_joint;
    as_of_date = obs.as_of_date);

#md # ```@raw html
#md # </details>
#md # ```

start_date_fig #hide

# The full posterior summary table reports equal-tailed 30%, 60% and
# 90% credible intervals on the key joint-fit parameters: growth rate
# $r$, doubling-time multiplier $m$, days since seeding $T$, CFR, the
# DRC and Uganda ascertainment fractions $p_{\text{DRC}}$ and $p_{\text{Uganda}}$, the
# pooling SD $\tau_{\text{logit}}$, the surveillance dispersion on both
# the sampled $1/\sqrt{k}$ scale and the more familiar $k$ scale, and
# cumulative cases $C(T)$.

#md # ```@raw html
#md # <details><summary>Joint posterior summary table</summary>
#md # ```

joint_summary = summary_table(chn_joint,
    [:r, :m, :T, :CFR, :p_drc, :p_uganda, :τ_logit,
     :inv_sqrt_k, :k, :cumulative_cases]; digits = 2);

#md # ```@raw html
#md # </details>
#md # ```

joint_summary #hide

# The posterior pair plot shows the joint distribution of the key
# parameters, with the prior overlaid so the data's contribution to
# each marginal is visible.

#md # ```@raw html
#md # <details><summary>Posterior pair plot (prior overlaid)</summary>
#md # ```

posterior_pair_fig = plot_pair(chn_joint,
    [:τ, :m, :cumulative_cases, :CFR, :w, :inv_sqrt_k, :k,
     :p_drc, :p_uganda, :τ_logit];
    prior = prior_chn);

#md # ```@raw html
#md # </details>
#md # ```

posterior_pair_fig #hide

# A posterior predictive check draws replicated observations from the
# fitted joint model and compares them to the observed counts. If the
# fit is reasonable the observed value (red line) sits inside the bulk
# of its replicate distribution. The four panels are the four data
# streams: exported cases and deaths among exports (Uganda), and deaths
# and reported cases (DRC).

#md # ```@raw html
#md # <details><summary>Joint posterior predictive plot</summary>
#md # ```

pp_joint   = predict(
    bvd_joint(missing, missing, missing,
              fill(missing, length(obs.export_deaths_daily));
              pre_start_deaths = missing,
              pre_detection_exports = missing,
              first_export_detection_delta = obs.first_export_detection_delta,
              genetic = genetic_seeding),
    chn_joint);
pp_exports        = vec(Array(pp_joint[:exported_cases]));
pp_deaths         = vec(Array(pp_joint[:total_deaths]));
pp_cases          = vec(Array(pp_joint[:reported_cases]));
## Export deaths are a per-day series held as one vector-valued variable
## in the FlexiChains predictive; sum each draw's daily counts for the
## total-count posterior predictive.
pp_exports_deaths = vec(sum.(pp_joint[@varname(export_deaths_daily)]));

joint_ppc_fig = plot_posterior_predictive(
    pp_exports, pp_deaths,
    obs.exported_cases, obs.total_deaths;
    pp_cases           = pp_cases,
    obs_cases          = obs.reported_cases,
    pp_exports_deaths  = pp_exports_deaths,
    obs_exports_deaths = obs.exports_deaths);

#md # ```@raw html
#md # </details>
#md # ```

joint_ppc_fig #hide

# ### Counterfactual: lower bound under no further transmission
#
# The lower bound on cumulative deaths if transmission stopped at the
# report date: still-expected and projected-total deaths per draw.

#md # ```@raw html
#md # <details><summary>Project no-onward deaths and summarise</summary>
#md # ```

no_onward = predict_no_onward_deaths(chn_joint; obs_deaths = obs.total_deaths);

no_onward_table = streams_table(
    "no-onward total" => no_onward.total_projected;
    digits = 0);

#md # ```@raw html
#md # </details>
#md # ```

no_onward_table #hide

# Two panels: the *still expected* deaths $\Delta D$ (future deaths in
# cases
# already infected by $T$, net of those already observed) on the left,
# and the *projected total* $D(T) + \Delta D$ on the right with the
# observed
# death count marked as a dashed black rule.

#md # ```@raw html
#md # <details><summary>No-onward projected-deaths plot</summary>
#md # ```

no_onward_fig = plot_no_onward_deaths(no_onward; obs_deaths = obs.total_deaths);

#md # ```@raw html
#md # </details>
#md # ```

no_onward_fig #hide

# ### One-week-ahead forecast
#
# The seven-day no-change projection: cumulative and new expected counts
# per stream by $T + 7$.

#md # ```@raw html
#md # <details><summary>Generate the one-week-ahead forecast</summary>
#md # ```

forecast = forecast_reported(chn_joint;
    horizon           = 7,
    daily_travellers  = ITURI_DAILY_TRAVEL,
    source_population = ITURI_POPULATION,
    obs_cases         = obs.reported_cases,
    obs_deaths        = obs.total_deaths,
    obs_exports       = EXPORTED_CASES);
forecast_summary = forecast_table(forecast);

#md # ```@raw html
#md # </details>
#md # ```

forecast_summary #hide

# New counts expected over the coming week, by stream:

#md # ```@raw html
#md # <details><summary>One-week-ahead forecast plot</summary>
#md # ```

forecast_fig = plot_forecast(forecast);

#md # ```@raw html
#md # </details>
#md # ```

forecast_fig #hide

# ### Delay sensitivity
#
# Refit under the community-only onset-to-death delay: the baseline and
# re-anchored $C(T)$ posteriors side by side.

#md # ```@raw html
#md # <details><summary>Refit the joint model with the community-only delay</summary>
#md # ```

community_delay = delay_model(;
    alpha_prior = truncated(Normal(5.6, (25.9 - 1.0) / 3.92); lower = 0),
    theta_prior = truncated(Normal(1.4, (9.5 - 0.3) / 3.92); lower = 0))

chn_joint_community = nuts_sample(
    bvd_joint(obs.exported_cases, obs.total_deaths,
              obs.reported_cases, obs.export_deaths_daily;
              first_export_detection_delta = obs.first_export_detection_delta,
              genetic = genetic_seeding,
              deaths = (total_deaths, growth_state, k) ->
                  deaths_model(total_deaths, growth_state, k;
                               delay = community_delay)));

posterior_C_community = vec(Array(chn_joint_community[:cumulative_cases]));

#md # ```@raw html
#md # </details>
#md # ```

# Fit diagnostics for the community-only delay refit.

#md # ```@raw html
#md # <details><summary>Fit diagnostics</summary>
#md # ```

diagnostics_table( #hide
    "joint (community-only delay)" => chn_joint_community) #hide

#md # ```@raw html
#md # </details>
#md # ```

# Baseline versus community-only delay, side by side:

#md # ```@raw html
#md # <details><summary>Delay-sensitivity C_T table</summary>
#md # ```

delay_sensitivity_table = streams_table(
    "joint (baseline delay)"        => posterior_C_joint,
    "joint (community-only delay)"  => posterior_C_community);

#md # ```@raw html
#md # </details>
#md # ```

delay_sensitivity_table #hide

# Overlaid posterior densities of $C(T)$ under the two delay
# assumptions:

#md # ```@raw html
#md # <details><summary>Delay-sensitivity C_T density plot</summary>
#md # ```

delay_sensitivity_fig = plot_cumulative_cases(
    "baseline delay"       => posterior_C_joint,
    "community-only delay" => posterior_C_community;
    scenarios = []);

#md # ```@raw html
#md # </details>
#md # ```

delay_sensitivity_fig #hide

# ### How the data streams compare
#
# Each data stream constrains the latent outbreak size differently.
# The table below puts the posteriors over $C(T)$ side by side — the
# four single-stream fits and the joint — to show what each stream buys
# on its own and what the joint combination adds.

#md # ```@raw html
#md # <details><summary>Per-stream C_T table</summary>
#md # ```

streams_C_table = streams_table(
    "exports (cases)" => posterior_C_exports,
    "exports (deaths)" => posterior_C_exports_deaths,
    "deaths (DRC)"  => posterior_C_deaths,
    "cases (DRC)"   => posterior_C_cases,
    "joint"        => posterior_C_joint);

#md # ```@raw html
#md # </details>
#md # ```

streams_C_table #hide

# The 2×4 grid below has replicates from the per-stream fits on the
# top row and the joint fit on the bottom row, comparable column-wise
# so it is easy to see what each per-stream fit constrains and how the
# joint combination shifts the predictives.

#md # ```@raw html
#md # <details><summary>Per-stream vs joint posterior predictive grid</summary>
#md # ```

pp_exports_only = vec(Array(predict(
    exports_only_model(missing), chn_exports)[:exported_cases]));
pp_deaths_only  = vec(Array(predict(
    deaths_only_model(missing),  chn_deaths )[:total_deaths]));
pp_cases_only   = vec(Array(predict(
    cases_only_model(missing),   chn_cases  )[:reported_cases]));
pp_exports_deaths_only = vec(sum.(predict(
    exports_deaths_only_model(fill(missing, length(obs.export_deaths_daily));
        pre_start_deaths = missing),
    chn_exports_deaths)[@varname(export_deaths_daily)]));

ppc_grid_fig = plot_posterior_predictive_grid(;
    individual = (; exports = pp_exports_only,
                    exports_deaths = pp_exports_deaths_only,
                    deaths  = pp_deaths_only,
                    cases   = pp_cases_only),
    joint      = (; exports = pp_exports,
                    exports_deaths = pp_exports_deaths,
                    deaths  = pp_deaths,
                    cases   = pp_cases),
    observed   = (; exports = obs.exported_cases,
                    exports_deaths = obs.exports_deaths,
                    deaths  = obs.total_deaths,
                    cases   = obs.reported_cases),
);

#md # ```@raw html
#md # </details>
#md # ```

ppc_grid_fig #hide

# Overlaid posterior densities of $C(T)$ from the four fits:

#md # ```@raw html
#md # <details><summary>Overlaid C_T density plot</summary>
#md # ```

cumulative_density_fig = plot_cumulative_cases(
    "exports (cases)" => posterior_C_exports,
    "exports (deaths)" => posterior_C_exports_deaths,
    "deaths (DRC)"  => posterior_C_deaths,
    "cases (DRC)"   => posterior_C_cases,
    "joint"        => posterior_C_joint;
    scenarios = []);

#md # ```@raw html
#md # </details>
#md # ```

cumulative_density_fig #hide

# ### Comparison with McCabe et al.
#
# Our joint fit against the McCabe et al. estimates and our Method 2
# reproduction: point estimates with 90% intervals. We step through
# both report versions in turn — the 18 May version [mccabe2026](@cite)
# and the 20 May version [mccabe2026update](@cite) — and end with our
# joint fit to the current data.

#md # ```@raw html
#md # <details><summary>Fit our model to each report version's data, and run the Method 2 reproductions</summary>
#md # ```

obs_report = load_observations(
    joinpath(pkgdir(BVDOutbreakSize), "data", "report-snapshot.toml"));
obs_report_20may = load_observations(
    joinpath(pkgdir(BVDOutbreakSize), "data",
             "report-snapshot-20may.toml"));

chn_joint_report = nuts_sample(
    bvd_joint(obs_report.exported_cases, obs_report.total_deaths,
              obs_report.reported_cases, obs_report.export_deaths_daily;
              first_export_detection_delta =
                  obs_report.first_export_detection_delta));
posterior_C_joint_report =
    vec(Array(chn_joint_report[:cumulative_cases]));

chn_joint_report_20may = nuts_sample(
    bvd_joint(obs_report_20may.exported_cases,
              obs_report_20may.total_deaths,
              obs_report_20may.reported_cases,
              obs_report_20may.export_deaths_daily;
              first_export_detection_delta =
                  obs_report_20may.first_export_detection_delta));
posterior_C_joint_report_20may =
    vec(Array(chn_joint_report_20may[:cumulative_cases]));

imperial_fixed = Turing.fix(
    imperial_only_model(missing, 88),       # exports missing → pure Method 2
    (τ = 14.0, CFR = 0.30, α = 4.42, θ = 1/0.388,
     inv_sqrt_k = 1e-3),
)
chn_imperial = nuts_sample(imperial_fixed);
posterior_C_imperial = vec(Array(chn_imperial[:cumulative_cases]));

imperial_fixed_20may = Turing.fix(
    imperial_only_model(missing, 131),      # exports missing → pure Method 2
    (τ = 14.0, CFR = 0.33, α = 4.42, θ = 1/0.388,
     inv_sqrt_k = 1e-3),
)
chn_imperial_20may = nuts_sample(imperial_fixed_20may);
posterior_C_imperial_20may =
    vec(Array(chn_imperial_20may[:cumulative_cases]));

#md # ```@raw html
#md # </details>
#md # ```

# The plot places each estimate of $C(T)$ on one axis: the central
# estimate as a point, the 90% interval as a bar. Rows are grouped by
# report version. For each version the first two rows are McCabe et
# al.'s published Method 1 and Method 2 headline scenarios with their
# reported intervals, followed by our Method 2 reproduction and our
# joint fit to that version's data. The final row is our joint fit to
# the current data.

#md # ```@raw html
#md # <details><summary>Build the comparison</summary>
#md # ```

joint_C_credibles        = posterior_summary(posterior_C_joint)
joint_report_C_credibles = posterior_summary(posterior_C_joint_report)
joint_report_20may_C_credibles =
    posterior_summary(posterior_C_joint_report_20may)
imperial_C_credibles     = posterior_summary(posterior_C_imperial)
imperial_20may_C_credibles =
    posterior_summary(posterior_C_imperial_20may)

comparison_rows = [
    ("18 May: McCabe Method 1 (Ituri, w=15 d)",   313, 39, 870),
    ("18 May: McCabe Method 2 (τ=14 d, CFR 30%)", 501, 402, 612),
    ("18 May: Our Method 2 reproduction",
        round(Int, quantile(posterior_C_imperial, 0.5)),
        round(Int, imperial_C_credibles.lo90),
        round(Int, imperial_C_credibles.hi90)),
    ("18 May: Our joint (report data)",
        round(Int, quantile(posterior_C_joint_report, 0.5)),
        round(Int, joint_report_C_credibles.lo90),
        round(Int, joint_report_C_credibles.hi90)),
    ("20 May: McCabe Method 1 (Ituri, w=15 d)",   313, 39, 870),
    ("20 May: McCabe Method 2 (τ=14 d, CFR 33%)", 678, 568, 800),
    ("20 May: Our Method 2 reproduction",
        round(Int, quantile(posterior_C_imperial_20may, 0.5)),
        round(Int, imperial_20may_C_credibles.lo90),
        round(Int, imperial_20may_C_credibles.hi90)),
    ("20 May: Our joint (report data)",
        round(Int, quantile(posterior_C_joint_report_20may, 0.5)),
        round(Int, joint_report_20may_C_credibles.lo90),
        round(Int, joint_report_20may_C_credibles.hi90)),
    ("Our joint (current data)",
        round(Int, quantile(posterior_C_joint, 0.5)),
        round(Int, joint_C_credibles.lo90),
        round(Int, joint_C_credibles.hi90)),
]

comparison_fig = plot_estimate_comparison(comparison_rows);

#md # ```@raw html
#md # </details>
#md # ```

comparison_fig #hide

# Fit diagnostics for the two report-data joint fits and the two Method
# 2 reproductions.

#md # ```@raw html
#md # <details><summary>Fit diagnostics</summary>
#md # ```

diagnostics_table( #hide
    "joint (18 May report)" => chn_joint_report, #hide
    "joint (20 May report)" => chn_joint_report_20may, #hide
    "Method 2 reproduction (18 May)" => chn_imperial, #hide
    "Method 2 reproduction (20 May)" => chn_imperial_20may) #hide

#md # ```@raw html
#md # </details>
#md # ```

# The same comparison as a table, with a column for the report version:

#md # ```@raw html
#md # <details><summary>Comparison table</summary>
#md # ```

comparison_version = [
    "18 May", "18 May", "18 May", "18 May",
    "20 May", "20 May", "20 May", "20 May",
    "current",
]

main_comparison = DataFrame(
    "Report version"   => comparison_version,
    "Source"           => [r[1] for r in comparison_rows],
    "Central estimate" => [r[2] for r in comparison_rows],
    "Lower 90%"        => [r[3] for r in comparison_rows],
    "Upper 90%"        => [r[4] for r in comparison_rows],
);

#md # ```@raw html
#md # </details>
#md # ```

main_comparison #hide

# Joint posterior coverage of all 15 published McCabe et al. scenarios
# — for each scenario, the narrowest joint credible interval that
# contains it:

#md # ```@raw html
#md # <details><summary>Joint coverage table</summary>
#md # ```

coverage_table = comparison_table(posterior_C_joint);

#md # ```@raw html
#md # </details>
#md # ```

coverage_table #hide

# The joint $C(T)$ density with the 15 published scenario point
# estimates overlaid as faint dashed rules, for our current-data fit
# and our fits to each report version's data:

#md # ```@raw html
#md # <details><summary>Joint C_T density with published scenarios</summary>
#md # ```

imperial_density_fig = plot_cumulative_cases(
    "joint (current data)"   => posterior_C_joint,
    "joint (18 May report)"  => posterior_C_joint_report,
    "joint (20 May report)"  => posterior_C_joint_report_20may);

#md # ```@raw html
#md # </details>
#md # ```

imperial_density_fig #hide

# ### McCabe et al. report sense check
#
# Whether each reproduction lands on McCabe et al.'s reported Method 2
# central estimate: the 18 May reproduction against their reported
# $501$ ($88$ deaths, CFR $30\%$) and the 20 May reproduction against
# their reported $678$ ($131$ deaths, CFR $33\%$).

#md # ```@raw html
#md # <details><summary>Reproductions vs McCabe et al. Method 2</summary>
#md # ```

imperial_sense_check = let
    rep = round(Int, quantile(posterior_C_imperial, 0.5))
    lo  = round(Int, imperial_C_credibles.lo90)
    hi  = round(Int, imperial_C_credibles.hi90)
    delta = round(100 * (rep - 501) / 501; digits = 1)
    rep2 = round(Int, quantile(posterior_C_imperial_20may, 0.5))
    lo2  = round(Int, imperial_20may_C_credibles.lo90)
    hi2  = round(Int, imperial_20may_C_credibles.hi90)
    delta2 = round(100 * (rep2 - 678) / 678; digits = 1)
    Markdown.parse("""
    18 May reproduction: **$(rep) cases** (90% CrI $(lo)–$(hi)) against
    McCabe et al.'s reported **501** — a difference of $(delta)%.

    20 May reproduction: **$(rep2) cases** (90% CrI $(lo2)–$(hi2))
    against McCabe et al.'s reported **678** — a difference of
    $(delta2)%.
    """)
end;

#md # ```@raw html
#md # </details>
#md # ```

imperial_sense_check #hide

#md # ```@raw html
#md # <details><summary>Sense-check summary tables</summary>
#md # ```

imperial_summary = summary_table(chn_imperial,
    [:m, :T, :cumulative_cases]; digits = 1);
imperial_summary_20may = summary_table(chn_imperial_20may,
    [:m, :T, :cumulative_cases]; digits = 1);

#md # ```@raw html
#md # </details>
#md # ```

imperial_summary #hide

imperial_summary_20may #hide

# ## Saving results
#
# The tables above are written to an `output/` directory at the repo
# root so they can be archived and shared. On every push to `main` a
# GitHub Actions workflow regenerates these files and publishes them
# as a GitHub Release, downloadable from the repository's releases
# page (<https://github.com/epiforecasts/BVDOutbreakSize/releases>).
# The release bundles the four summary tables, a thinned set of
# posterior draws, and a copy of the input `observations.toml` so the
# exact data that produced each result is recorded alongside it.

#md # ```@raw html
#md # <details><summary>Write outputs to output/</summary>
#md # ```

## Outputs default to `output/` in the package directory (where the
## docs build and Release workflow expect them). Set `BVD_OUTPUT_DIR`
## to redirect them, e.g. when running from a read-only package
## install.
output_dir = get(ENV, "BVD_OUTPUT_DIR",
                 joinpath(pkgdir(BVDOutbreakSize), "output"))
mkpath(output_dir)

CSV.write(joinpath(output_dir, "posterior_summary.csv"), joint_summary)
CSV.write(joinpath(output_dir, "cumulative_cases_by_stream.csv"),
          streams_C_table)
CSV.write(joinpath(output_dir, "imperial_comparison.csv"), main_comparison)
CSV.write(joinpath(output_dir, "scenario_coverage.csv"), coverage_table)

## Copy the input data so the release records what produced these
## results.
cp(joinpath(pkgdir(BVDOutbreakSize), "data", "observations.toml"),
   joinpath(output_dir, "observations.toml"); force = true)

## Thinned posterior draws of the key joint parameters (every 10th
## draw) so downstream users can recompute their own summaries.
posterior_draws = DataFrame(
    τ                = vec(Array(chn_joint[:τ])),
    r                = vec(Array(chn_joint[:r])),
    m                = vec(Array(chn_joint[:m])),
    T                = vec(Array(chn_joint[:T])),
    CFR              = vec(Array(chn_joint[:CFR])),
    p_drc            = vec(Array(chn_joint[:p_drc])),
    p_uganda         = vec(Array(chn_joint[:p_uganda])),
    cumulative_cases = vec(Array(chn_joint[:cumulative_cases])),
)[1:10:end, :]
CSV.write(joinpath(output_dir, "posterior_draws.csv"), posterior_draws);

#md # ```@raw html
#md # </details>
#md # ```

# ---
#
# The full analysis code, data and model definitions are in the
# [epiforecasts/BVDOutbreakSize](https://github.com/epiforecasts/BVDOutbreakSize)
# repository. Issues, corrections and suggestions are welcome there.
# Maintained by Sam Abbott, Kath Sherratt, Samuel Brand and Sebastian
# Funk.
