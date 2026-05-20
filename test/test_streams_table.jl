## Tests for streams_table: one row per stream with the documented
## column order.

@testset "streams_table returns expected columns and row count" begin
    rng = MersenneTwister(1)
    a = randn(rng, 500) .* 50 .+ 400
    b = randn(rng, 500) .* 80 .+ 600

    df = streams_table("fit A" => a, "fit B" => b)

    @test df isa DataFrame
    @test names(df) ==
          ["stream", "lower_90", "lower_60", "lower_30",
           "upper_30", "upper_60", "upper_90"]
    @test nrow(df) == 2
    @test df.stream == ["fit A", "fit B"]

    for r in eachrow(df)
        @test r.lower_90 <= r.lower_60 <= r.lower_30 <=
              r.upper_30 <= r.upper_60 <= r.upper_90
    end
end
