## Tests for default_adtype: returns an AutoMooncake backend.

@testitem "default_adtype returns an AutoMooncake" begin
    using ADTypes: AutoMooncake
    using BVDOutbreakSize: default_adtype
    ad = default_adtype()
    @test ad isa AutoMooncake
end
