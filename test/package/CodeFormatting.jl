@testitem "Code formatting" tags=[:quality] begin
    using Pkg
    formatter_env = joinpath(@__DIR__, "..", "formatter")
    if isdir(formatter_env) && isfile(joinpath(formatter_env, "Project.toml"))
        # Instantiate the formatter environment via a subprocess so the
        # active project of the test process is not mutated (otherwise
        # later @testitems lose access to BVDOutbreakSize).
        run(`julia --project=$formatter_env -e "using Pkg; Pkg.instantiate()"`)
        cmd = Cmd(
            `julia --project=$formatter_env $(joinpath(formatter_env, "runtests.jl"))`;
            ignorestatus = true)
        result = run(pipeline(cmd, stdout = stdout, stderr = stderr); wait = true)
        @test result.exitcode == 0
    else
        @test_skip "Formatter environment not found"
    end
end
