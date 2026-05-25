using Test
import BVDOutbreakSize
using BVDOutbreakSize: REPORT_SCENARIOS, expected_deaths,
                       summary_table, posterior_summary,
                       fit_diagnostics, diagnostics_table,
                       streams_table, comparison_table,
                       nuts_sample, default_adtype,
                       load_observations,
                       plot_cumulative_cases, plot_density_overlay,
                       plot_prior_predictive,
                       plot_posterior_predictive, plot_pair,
                       forecast_reported, forecast_table, plot_forecast,
                       predict_no_onward_deaths, plot_no_onward_deaths
                       plot_start_date_pair, plot_estimate_comparison,
                       forecast_reported, forecast_table, plot_forecast,
                       forecast_vs_truth, plot_forecast_vs_truth
using ADTypes: AutoMooncake
import CairoMakie
using DataFrames: DataFrame, nrow
using Distributions: Beta, Gamma, NegativeBinomial, Normal, Poisson
using Distributions: pdf, truncated
using Integrals: IntegralProblem, GaussLegendre, solve
using JET: test_opt
using Mooncake: Mooncake
using Random: MersenneTwister
using Statistics: quantile
using StatsFuns: logit, logistic
using Turing: Turing, @model, sample, Prior, to_submodel
import MCMCChains
using MCMCChains: Chains
import FlexiChains
using FlexiChains: VNChain
import CairoMakie

# Make sure Makie does not try to open a screen.
CairoMakie.activate!(type = "png")

include("test_posterior_summary.jl")
include("test_summary_table.jl")
include("test_streams_table.jl")
include("test_comparison_table.jl")
include("test_plots.jl")
include("test_adtype.jl")
include("test_nuts_sample.jl")
include("test_load_observations.jl")
include("test_genetic_seeding.jl")
include("test_diagnostics.jl")
include("test_no_onward_deaths.jl")
include("test_cases_model.jl")
include("test_forecast.jl")
include("test_pooled_ascertainment.jl")
include("test_exports_deaths.jl")
include("test_expected_deaths.jl")
include("test_integrate.jl")
include("test_exports_death_timing.jl")
include("test_exports_delay_grid.jl")
