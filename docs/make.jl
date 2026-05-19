using Pkg: Pkg
Pkg.instantiate()

using Documenter
using DocumenterCitations
using Literate

const bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :authoryear,
)

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

# References page sourced from refs.bib through `@bibliography`.
open(joinpath(LITERATE_OUT, "references.md"), "w") do io
    println(io, "# References")
    println(io)
    println(io, "```@bibliography")
    println(io, "```")
end

makedocs(;
    sitename = "BVDOutbreakSize",
    authors  = "Sam Abbott and contributors",
    repo     = "github.com/seabbs/BVDOutbreakSize",
    clean    = true,
    doctest  = false,
    warnonly = [:missing_docs, :linkcheck, :citations],
    plugins  = [bib],
    pages    = [
        "Home"                 => "index.md",
        "Analysis walkthrough" => "analysis.md",
        "References"           => "references.md",
    ],
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://seabbs.github.io/BVDOutbreakSize",
        size_threshold        = 1_000_000,
        size_threshold_warn   = 800_000,
    ),
)

deploydocs(;
    repo        = "github.com/seabbs/BVDOutbreakSize",
    target      = "build",
    branch      = "gh-pages",
    devbranch   = "main",
    push_preview = true,
)
