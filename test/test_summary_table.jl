## Tests for summary_table: builds a DataFrame with one row per
## parameter and the documented quantile columns. We sample from
## Prior() on a trivial Turing model so the test does not depend on
## NUTS warm-up.

@model function _summary_model()
    a ~ Normal(0.0, 1.0)
    b ~ Normal(2.0, 0.5)
end

@testset "summary_table returns expected columns and rows" begin
    chn = sample(_summary_model(), Prior(), 200;
                 chain_type = MCMCChains.Chains, progress = false)
    params = [:a, :b]
    df = summary_table(chn, params)

    @test df isa DataFrame
    @test names(df) ==
          ["quantity", "lower_90", "lower_60", "lower_30",
           "upper_30", "upper_60", "upper_90"]
    @test nrow(df) == length(params)
    @test df.quantity == ["a", "b"]

    # Each row's quantile columns are monotone (an internal sanity
    # check that the rounded entries still respect the ordering).
    for r in eachrow(df)
        @test r.lower_90 <= r.lower_60 <= r.lower_30
        @test r.upper_30 <= r.upper_60 <= r.upper_90
        @test r.lower_30 <= r.upper_30
    end
end
