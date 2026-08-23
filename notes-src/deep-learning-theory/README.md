# Deep learning theory notes

LaTeX notes on kernels, infinite width, and effective theory.

## Compile

```bash
cd notes-src/deep-learning-theory
python scripts/generate_figures.py          # optional; PDFs already in figures/
latexmk -pdf -outdir=build main.tex
# or:
tectonic -X compile --outdir build main.tex
```

Zip this directory with `main.tex` at the root for Overleaf (`figures/` included; Python not required there).
