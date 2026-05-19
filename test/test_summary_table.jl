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
          ["quantity", "lo90", "lo60", "lo30", "hi30", "hi60", "hi90"]
    @test nrow(df) == length(params)
    @test df.quantity == ["a", "b"]

    # Each row's quantile columns are monotone (an internal sanity
    # check that the rounded entries still respect the ordering).
    for r in eachrow(df)
        @test r.lo90 <= r.lo60 <= r.lo30
        @test r.hi30 <= r.hi60 <= r.hi90
        @test r.lo30 <= r.hi30
    end
end
