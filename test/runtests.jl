using Test
import BVDOutbreakSize
using BVDOutbreakSize: REPORT_SCENARIOS,
                       summary_table, posterior_summary,
                       fit_diagnostics, diagnostics_table,
                       streams_table, comparison_table,
                       nuts_sample, default_adtype,
                       load_observations,
                       plot_cumulative_cases,
                       plot_prior_predictive,
                       plot_posterior_predictive, plot_pair,
                       forecast_reported, forecast_table, plot_forecast
using ADTypes: AutoMooncake
using DataFrames: DataFrame, nrow
using Distributions: Normal
using Random: MersenneTwister
using Turing: Turing, @model, sample, Prior
import MCMCChains
using MCMCChains: Chains
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
include("test_diagnostics.jl")
include("test_no_onward_deaths.jl")
include("test_cases_model.jl")
include("test_forecast.jl")
include("test_pooled_ascertainment.jl")
include("test_exports_deaths.jl")
include("test_expected_deaths.jl")
