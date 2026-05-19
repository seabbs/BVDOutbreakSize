using Pkg: Pkg
Pkg.instantiate()

using Documenter
using Literate

const REPO_ROOT    = dirname(@__DIR__)
const LITERATE_SRC = joinpath(@__DIR__, "examples", "analysis.jl")
const LITERATE_OUT = joinpath(@__DIR__, "src")

isdir(LITERATE_OUT) || mkpath(LITERATE_OUT)

cp(joinpath(REPO_ROOT, "README.md"),
   joinpath(LITERATE_OUT, "index.md");
   force = true)

Literate.markdown(
    LITERATE_SRC,
    LITERATE_OUT;
    name    = "analysis",
    flavor  = Literate.DocumenterFlavor(),
    execute = true,
    credit  = false,
)

makedocs(;
    sitename = "BVDOutbreakSize",
    authors  = "Sam Abbott and contributors",
    clean    = true,
    doctest  = false,
    warnonly = [:missing_docs, :linkcheck],
    pages    = [
        "Home"                 => "index.md",
        "Analysis walkthrough" => "analysis.md",
    ],
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://seabbs.github.io/BVDOutbreakSize",
    ),
)

deploydocs(;
    repo        = "github.com/seabbs/BVDOutbreakSize",
    target      = "build",
    branch      = "gh-pages",
    devbranch   = "main",
    push_preview = true,
)
