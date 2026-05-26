using JuliaFormatter

project_root = dirname(dirname(@__DIR__))
dirs_to_check = filter(isdir,
    [joinpath(project_root, d) for d in ("src", "test", "docs", "scripts")])
all_formatted = all(d -> JuliaFormatter.format(d; verbose = true, overwrite = false),
    dirs_to_check)
exit(all_formatted ? 0 : 1)
