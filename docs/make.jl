using Pkg: Pkg
Pkg.instantiate()

using Documenter
using DocumenterCitations
using DocumenterVitepress
using Literate

const bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :authoryear,
)

const REPO_ROOT    = dirname(@__DIR__)
const LITERATE_SRC = joinpath(@__DIR__, "examples", "analysis.jl")
const LITERATE_OUT = joinpath(@__DIR__, "src")

isdir(LITERATE_OUT) || mkpath(LITERATE_OUT)

# Copy the README to the home page, stripping the ABSTRACT marker
# comments. The analysis page reads them from the source README to load
# the abstract, but they must not appear on the rendered home page (the
# Vitepress typographer mangles the `--` and shows them as text).
let readme = read(joinpath(REPO_ROOT, "README.md"), String)
    readme = replace(readme, r"^<!-- ABSTRACT:(START|END) -->\n"m => "")
    write(joinpath(LITERATE_OUT, "index.md"), readme)
end

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
    repo     = "github.com/epiforecasts/BVDOutbreakSize",
    clean    = true,
    doctest  = false,
    warnonly = [:missing_docs, :linkcheck, :citations],
    plugins  = [bib],
    pages    = [
        "Home"        => "index.md",
        "Analysis"    => "analysis.md",
        "News"        => "news.md",
        "References"  => "references.md",
    ],
    format   = DocumenterVitepress.MarkdownVitepress(;
        repo      = "github.com/epiforecasts/BVDOutbreakSize",
        devbranch = "main",
        devurl    = "dev",
    ),
)

# Use DocumenterVitepress.deploydocs, not the bare Documenter one:
# DocumenterVitepress 0.2 builds into numbered subfolders
# (docs/build/1/, …) and its deploydocs flattens each build/i/ to
# gh-pages/<base>/. Plain deploydocs leaves the numbered subdir, so
# the deployed site's asset URLs 404. Ref LuxDL/DocumenterVitepress.jl#280.
DocumenterVitepress.deploydocs(;
    repo        = "github.com/epiforecasts/BVDOutbreakSize",
    target      = "build",
    branch      = "gh-pages",
    devbranch   = "main",
    push_preview = true,
)
