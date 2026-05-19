## Tests for default_adtype: returns an AutoMooncake backend.

@testset "default_adtype returns an AutoMooncake" begin
    ad = default_adtype()
    @test ad isa AutoMooncake
end
