.PHONY: build axioms mdgen html pdf clean all

PANDOC_FLAGS = --standalone --toc --mathjax \
    --css=style.css \
    --metadata title="Formal Verification of BVDOutbreakSize.jl" \
    --metadata author="Simon Frost" \
    --highlight-style=tango

all: build mdgen html

# Build the Lean 4 project (verifies all proofs)
build:
	lake build

# Print axiom dependencies for headline theorems
axioms:
	lake env lean AxiomReport.lean

# Generate markdown from Lean source via mdgen
mdgen:
	rm -rf .mdgen_tmp
	mkdir -p .mdgen_tmp
	lake exe mdgen BvdProofs .mdgen_tmp
	cat .mdgen_tmp/Foundations.md .mdgen_tmp/Quadrature.md \
	    .mdgen_tmp/Expectations.md .mdgen_tmp/GammaCDF.md \
	    .mdgen_tmp/Interpolation.md .mdgen_tmp/Forecast.md \
	    > BvdProofs.md
	rm -rf .mdgen_tmp
	@echo "Generated BvdProofs.md"

# Generate HTML from markdown
html: BvdProofs.md
	pandoc BvdProofs.md -o BvdProofs.html $(PANDOC_FLAGS)
	@echo "Generated BvdProofs.html"

# Generate PDF from markdown
pdf: BvdProofs.md
	pandoc BvdProofs.md -o BvdProofs.pdf \
	    --pdf-engine=lualatex \
	    --toc \
	    --metadata title="Formal Verification of BVDOutbreakSize.jl" \
	    --metadata author="Simon Frost" \
	    --highlight-style=tango \
	    -H header.tex
	@echo "Generated BvdProofs.pdf"

clean:
	rm -f BvdProofs.html BvdProofs.pdf BvdProofs.md
	rm -rf .mdgen_tmp
