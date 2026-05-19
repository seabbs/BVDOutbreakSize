## Tests for streams_table: one row per stream with the documented
## column order.

@testset "streams_table returns expected columns and row count" begin
    rng = MersenneTwister(1)
    a = randn(rng, 500) .* 50 .+ 400
    b = randn(rng, 500) .* 80 .+ 600

    df = streams_table("fit A" => a, "fit B" => b)

    @test df isa DataFrame
    @test names(df) ==
          ["stream", "lo90", "lo60", "lo30", "hi30", "hi60", "hi90"]
    @test nrow(df) == 2
    @test df.stream == ["fit A", "fit B"]

    for r in eachrow(df)
        @test r.lo90 <= r.lo60 <= r.lo30 <= r.hi30 <= r.hi60 <= r.hi90
    end
end
