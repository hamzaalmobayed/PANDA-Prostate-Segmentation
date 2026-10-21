# PANDA Reproducibility Changelog

## Reproducibility package update - October 2026

### Results and manifests
- Added six B1-B4 statistical and common-support CSV exports.
- Added `selected/train_selected.csv` (80,000 selected patches manifest).
- Existing split, training, evaluation, and model-weight materials
  should be checked against the final manuscript.

### Environment
- Added pinned `requirements.txt` from the experiment
  environment, including SciPy, as supplied and confirmed
  by the project owner.

### Documentation
- Added or updated the reproducibility README.
- Added an index of B1-B4 outputs.
- Added figure source-candidate and SHA-256 inventory.
- Added audit of nonportable absolute paths.

### Figure provenance caveat
Manual composition in Paint is compatible with preparing
figures, but source case IDs, ground-truth masks, original
model outputs and normalization parameters must still be
verified for scientific image panels. Script filenames alone
do not establish provenance.

### Known outstanding issues
- Complete provenance verification for scientific figures.
- Parameterize hardcoded operating-system-specific paths.
- Review legacy review notes before removing them.
- Clean historical personal files from Git history using a
  separately backed-up and audited history rewrite.

### Scope
This changelog is an inventory of reproducibility work,
not a claim that all study figures have been independently
reproduced.

Zenodo publication is intentionally excluded from this pass.