# Build the manuscript

Install Tectonic. From this directory:

    tectonic --keep-intermediates main.tex
    cd si
    tectonic --keep-intermediates ESM_1.tex

The first command produces main.pdf. The second produces si/ESM_1.pdf;
the release assembler also places a copy at paper/ESM_1.pdf when that build succeeds.
Build the main paper before the ESM so external references can use main.aux.
The included PDFs are built from this staging snapshot.

The shipped artwork already suffices to build the paper. To regenerate data,
run the relevant figures/make_*.py listed in ../code/MANIFEST.md. These scripts
use both formulas and archived numerical outputs; they do not all launch new
simulations. make_num.py intentionally reproduces the original pilot figure.

To compile and check artwork:

    python3 -I figures/build_figs.py
    python3 -I figures/check_figs.py

These require Tectonic, Poppler tools (pdftops, pdfinfo, pdffonts), PyMuPDF and
the Python dependencies in ../code/requirements.txt. figure-spec.json records
the release's data hashes, table selections and analytic curves; no private
audit directory is required. make_w2.py is an alternative ring/bounds driver.
The strict sans-family styling criterion currently flags num-1, whose archived
artwork uses Nimbus Sans and Computer Modern math fonts. Its data checks pass;
this is an artwork-style qualification, not a numerical-data discrepancy.
Publisher class/style files retain their embedded licence notices.
