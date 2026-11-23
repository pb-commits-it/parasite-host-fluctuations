# Do several hosts stabilize a parasite? Residence times, host fluctuations and the reproduction number of complex life cycles

Paul E. Bergeron, independent researcher.

Code and generated numerical results accompanying the bioRxiv preprint. The bioRxiv DOI will be added after deposition.

- [paper/](paper/README.md): manuscript, Online Resource 1, artwork and figure generators.
- [lean/](lean/README.md): pinned Lean formalization, axiom probe and coverage inventory.
- [code/](code/MANIFEST.md): simulation engines, checks, experiments, case studies, data fetchers and compact results.

## Reproduce

Use Python 3.11 or later (packaging checks used Python 3.13.5). Install the numerical dependencies in code/requirements.txt. From this repository root:

    python3 -m pip install -r code/requirements.txt
    python3 -I code/checks/route_sum_vs_r0.py
    python3 -I code/checks/unequal_route_lengths.py

Follow paper/README.md for Tectonic and figures, lean/README.md for Lean, and code/MANIFEST.md for the script/output corresponding to each reported computation. Run case studies from their own directory; raw inputs must first be downloaded with the matching code/data/fetch scripts. Read each campaign's protocol and command-line options before starting long runs. Packaging validation does not re-execute the historical campaigns.

## Data availability

The following is transcribed from the current manuscript, including any unresolved qualifications:

No new field or experimental data were collected. The Cary Institute forest-food-web dataset analyzed in Section 9 is publicly available on Dryad at [dryad.d1c8046](https://doi.org/10.5061/dryad.d1c8046) under a CC0 1.0 waiver (reference keys: ostfeld20182019); the longer Cary Institute synthesis series is available at [caryinstitute.31872853.v2](https://doi.org/10.25390/caryinstitute.31872853.v2) under a CC BY 4.0 license (reference keys: ostfeld2026b) (see also (reference keys: ostfeld2026a), [caryinstitute.31915557.v1](https://doi.org/10.25390/caryinstitute.31915557.v1)); and the 2014–2022 Cary Institute holdout dataset is available at [caryinstitute.27871269.v1](https://doi.org/10.25390/caryinstitute.27871269.v1) under deposit metadata license CC BY-NC 4.0 (older embedded method notes carry a pre-publication notice, with the public deposit metadata, which give CC BY-NC 4.0, treated as controlling in our analysis) (reference keys: ladeau2025). The National Ecological Observatory Network (NEON) RELEASE-2026 data products used in Section 9 are Small mammal box trapping (DP1.10072.001, [a83h-tb34](https://doi.org/10.48443/a83h-tb34)), Tick pathogen status (DP1.10092.001, [n2yp-5a62](https://doi.org/10.48443/n2yp-5a62)), and Ticks sampled using drag cloths (DP1.10093.001, [5e20-3763](https://doi.org/10.48443/5e20-3763)), accessed 14 November 2026 (reference keys: neon2026a,neon2026b,neon2026c) (provisional site-months present in the downloaded bundle are not covered by the release DOIs; the DataCite records of these releases give a CC0 1.0 license). Generated numerical results and reproduction outputs are archived at [parasite-host-fluctuations](https://github.com/pb-commits-it/parasite-host-fluctuations).

Additional case-study and replay deposit DOIs are listed in [code/data/README.md](code/data/README.md) and paper/refs.bib. No raw observations are redistributed.

## Citation and licenses

Bergeron, Paul E. Do several hosts stabilize a parasite? Residence times, host fluctuations and the reproduction number of complex life cycles. bioRxiv preprint, 2026. DOI: to be added. See [CITATION.cff](CITATION.cff).

Original project code is MIT licensed ([LICENSE](LICENSE)). Manuscript text and original figures are CC BY 4.0 ([LICENSE-text](LICENSE-text)). Bundled publisher class/style files retain their embedded notices; dependency and source-data licenses are separate.
