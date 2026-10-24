# Report

The assignment report in the ACM Small format (`acmart` class with the
`nonacm` option, so there is no ACM, journal or copyright information).

| File | Contents |
|---|---|
| `report.pdf` | the compiled report |
| `main.tex` | the report source |
| `references.bib` | bibliography (ACM reference format) |
| `figures/*.pdf` | copies of five of the figures in `results/figures/`, produced by `python -m scripts.plot_results` |

## Compiling on Overleaf

1. Zip this `report/` folder (or use `report_overleaf.zip` in the repository root if it exists).
2. In Overleaf, choose **New Project → Upload Project** and upload the zip.
3. Check that the compiler is **pdfLaTeX** (Menu → Settings) and the main document is `main.tex`.
4. Click **Recompile**. BibTeX runs automatically.

## Compiling locally

With a TeX distribution that includes `acmart` (TeX Live or MiKTeX):

```bash
latexmk -pdf main.tex
```

## Updating the figures

After a new benchmark run, regenerate the figures and copy the five used in the report:

```bash
python -m scripts.plot_results
cp results/figures/{first_extract_min,extract_min_time,decrease_key_time,merge_time,dijkstra_time}.pdf report/figures/
```
