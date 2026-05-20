using Pkg: Pkg
Pkg.instantiate()

using Documenter
using DocumenterCitations
using DocumenterVitepress
using Literate
using TikzPictures

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

# Model-structure diagram. Compiled from TikZ to a standalone SVG with
# lualatex + dvisvgm (via TikzPictures), then inlined into the analysis
# page in place of the `{{MODEL_DIAGRAM}}` placeholder. Inlining avoids
# relying on Vitepress relative-asset copying.
const DIAGRAM_BODY = raw"""
\graph[layered layout, grow=down,
       level distance=18mm, sibling distance=8mm,
       nodes={draw,rounded corners,align=center,
              font=\footnotesize,inner sep=3pt},
       edges={->,>={Stealth},gray}]{
  G  [as={Growth\\$C(s)=e^{rs}$}];
  D  [as={Onset-to-death\\delay}];
  CFR[as={Case-fatality\\ratio}];
  W  [as={Detection\\window}];
  K  [as={Surveillance\\dispersion}];
  A  [as={Ascertainment}];
  OE [as={Exports\\Poisson}];
  OD [as={Deaths\\NegBinomial}];
  OC [as={Cases\\NegBinomial}];
  OX [as={Exports-deaths\\Poisson}];
  CE [as={exports\_only}];
  CD [as={deaths\_only}];
  CC [as={cases\_only}];
  CX [as={exports\_deaths\_only}];
  CI [as={imperial\_only}];
  CJ [as={bvd\_joint}];
  G -> { OE, OD, OC, OX };
  D -> { OD, OX };
  CFR -> { OD, OX };
  W -> { OE, OX };
  K -> { OD, OC };
  A -> { OE, OC, OX };
  OE -> { CE, CI, CJ };
  OD -> { CD, CI, CJ };
  OC -> { CC, CJ };
  OX -> { CX, CJ };
};
"""

function model_diagram_svg()
    tp = TikzPicture(
        DIAGRAM_BODY;
        preamble = "\\usetikzlibrary{graphs,graphdrawing,arrows.meta}\n" *
                   "\\usegdlibrary{layered}",
    )
    out = joinpath(tempdir(), "model_structure")
    save(SVG(out), tp)
    svg = read(out * ".svg", String)
    svg = svg[findfirst("<svg", svg)[1]:end]            # drop xml/doctype
    style = "max-width:100%;height:auto;"
    svg = replace(svg, r"<svg " => "<svg style=\"$style\" "; count = 1)
    return "```@raw html\n<div style=\"text-align:center\">\n$svg\n</div>\n```"
end

let analysis_md = joinpath(LITERATE_OUT, "analysis.md")
    text = read(analysis_md, String)
    text = replace(text, "{{MODEL_DIAGRAM}}" => model_diagram_svg())
    write(analysis_md, text)
end

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
