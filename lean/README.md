# Lean formalization

Install the version of Lean specified by lean-toolchain with elan. From this directory:

    lake exe cache get
    lake build
    lake build DobsonProofs.P1bB_ExactR5 DobsonProofs.P1bB_ExactR6 DobsonProofs.P1bB_MechBounded
    lake env lean -DautoImplicit=false CheckAxioms.lean

The last command checks and prints the axioms of every credited declaration in the
2026-11-22 probe, including supplemental modules outside the umbrella import.
Expected dependencies are subsets of propext, Classical.choice and Quot.sound.
Any sorryAx or additional custom axiom requires investigation.
A supplementary source check is:

    rg -n '\\b(sorry|admit|axiom)\\b' DobsonProofs/ DobsonProofs.lean

Read comments separately from declarations. A successful build or clean axiom
report does not prove that a statement matches the manuscript.

[COVERAGE_INVENTORY.md](COVERAGE_INVENTORY.md) is generated from the version 2
inventory dated 2026-11-22. Current totals: 82 = 9 full + 57 core + 16 partial-P1.
Historical counts in its dated version 1 narrative are explicitly superseded.

- Full: the entire manuscript statement is checked under stated textbook or model assumptions.
- Core: the identified component is checked; listed omissions remain outside coverage.
- Partial: an identified component still assumes a paper-specific step.
  P1 has an independently checked paper proof; P2 lacks that independent check;
  P3 has no located proof.

Historical internal audit records are described with their dates, without links
to files absent from this repository. Source line anchors are historical locations;
use declaration names and manuscript labels if later comments shift line numbers.
