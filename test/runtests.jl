using Test
import BVDOutbreakSize
using BVDOutbreakSize: REPORT_SCENARIOS,
                       summary_table, posterior_summary,
                       streams_table, comparison_table,
                       nuts_sample, default_adtype,
                       plot_cumulative_cases,
                       plot_posterior_predictive, plot_pair
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
