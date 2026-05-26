## Tests for summary_table: builds a DataFrame with one row per
## parameter and the documented quantile columns. We sample from
## Prior() on a trivial Turing model so the test does not depend on
## NUTS warm-up.

@testitem "summary_table returns expected columns and rows" tags=[:slow] begin
    using DataFrames: DataFrame, nrow
    using Distributions: Normal
    using Turing: @model, sample, Prior
    import FlexiChains
    using BVDOutbreakSize: summary_table

    ## kept: summary_table only needs two named parameters with sensible
    ## quantiles; the real models drag in BVD-specific structure that the
    ## test does not need.
    @model function _summary_model()
        a ~ Normal(0.0, 1.0)
        b ~ Normal(2.0, 0.5)
    end

    chn = sample(_summary_model(), Prior(), 200;
                 chain_type = FlexiChains.VNChain, progress = false)
    params = [:a, :b]
    df = summary_table(chn, params)

    @test df isa DataFrame
    @test names(df) ==
          ["Quantity", "Lower 90%", "Lower 60%", "Lower 30%",
           "Upper 30%", "Upper 60%", "Upper 90%"]
    @test nrow(df) == length(params)
    @test df[!, "Quantity"] == ["a", "b"]

    # Each row's quantile columns are monotone (an internal sanity
    # check that the rounded entries still respect the ordering).
    for r in eachrow(df)
        @test r["Lower 90%"] <= r["Lower 60%"] <= r["Lower 30%"]
        @test r["Upper 30%"] <= r["Upper 60%"] <= r["Upper 90%"]
        @test r["Lower 30%"] <= r["Upper 30%"]
    end
end
