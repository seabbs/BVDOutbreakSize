module BVDOutbreakSize

using Statistics: quantile, mean, std
using Printf: @sprintf
using TOML
using DataFrames: DataFrame
using DataFramesMeta
using Chain: @chain
using Random: MersenneTwister
using Dates: Date, date2epochdays, epochdays2date
using ADTypes: AutoMooncake
using Mooncake: Mooncake
using ChainRulesCore: ChainRulesCore, NoTangent
using SpecialFunctions: gamma_inc, digamma, loggamma
import SpecialFunctions
using Turing
using Turing.DynamicPPL: InitFromPrior
import FlexiChains
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF,
                           TYPEDFIELDS, TYPEDSIGNATURES
using Distributions: Distribution, Gamma, cdf, ccdf, mgf, pdf, Poisson, NegativeBinomial
using Integrals: IntegralProblem, GaussLegendre, QuadGKJL, solve
import FastGaussQuadrature
import CairoMakie
import AlgebraOfGraphics as AoG
import PairPlots
using CairoMakie: Figure, Axis, hist!, density!, vlines!, vspan!,
                  lines!, scatter!

export REPORT_SCENARIOS,
       ITURI_POPULATION, ITURI_DAILY_TRAVEL,
       ITURI_DAILY_TRAVEL_SD,
       load_observations,
       summary_table, posterior_summary,
       fit_diagnostics, diagnostics_table,
       streams_table, comparison_table,
       nuts_sample, default_adtype,
       DEATH_INTEGRAL_ALG, CUMULATIVE_INTEGRAL_ALG,
       integrate, expected_deaths,
       integrate_cumulative, integrate_exports_deaths,
       expected_exports, expected_exports_deaths,
       ExportDeathDelay, EXPORT_DELAY_GRID_POINTS,
       plot_cumulative_cases, plot_density_overlay, plot_prior_predictive,
       plot_posterior_predictive, plot_posterior_predictive_grid,
       plot_pair, plot_start_date_pair, plot_estimate_comparison,
       plot_cfr_prior,
       predict_no_onward_deaths, plot_no_onward_deaths,
       forecast_reported, forecast_table, plot_forecast,
       forecast_vs_truth, plot_forecast_vs_truth

include("docstrings.jl")
include("constants.jl")
include("data.jl")
include("sampling.jl")
include("gamma_cdf.jl")
include("integrate.jl")
include("expectations.jl")
include("summaries.jl")
include("counterfactual.jl")
include("forecast.jl")
include("plots.jl")

end # module
