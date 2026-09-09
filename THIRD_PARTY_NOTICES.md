# Third-party sources and notices

This release is based on [AdaptCLIP](https://github.com/gaobb/AdaptCLIP).
The upstream GNU General Public License, version 2, is preserved in [LICENSE](LICENSE).
Upstream copyright and license notices remain applicable to their respective
portions of the code. This document records provenance; it does not replace or
expand any license grant.

## AdaptCLIP

- Source: <https://github.com/gaobb/AdaptCLIP>
- License source: <https://github.com/gaobb/AdaptCLIP/blob/main/LICENSE>
- Reference: Bin-Bin Gao et al., *AdaptCLIP: Adapting CLIP for Universal Visual
  Anomaly Detection*, AAAI 2026.

The model framework, training and evaluation scaffolding, dataset loaders, and
supporting utilities originate from AdaptCLIP and subsequent ReAxis research
changes. ReAxis adds its context, axis-reorientation, calibration, and fusion
modules. Public-release modifications dated 2026-09-09 include portable entry
points and documentation, project terminology, and the replacement loss and
evaluation implementations described below. See the affected source files for
modification notices.

The preserved upstream license contains the standard GPL v2 application example.
Its example wording is not an additional project-specific grant of an option to
use later GPL versions. This release does not claim such an upstream grant or
relicense upstream code under MIT or Apache-2.0.

## OpenAI CLIP

- Source: <https://github.com/openai/CLIP>
- License: [MIT license and copyright notice](licenses/OpenAI-CLIP-MIT.txt)
- Copyright (c) 2021 OpenAI

The CLIP backbone implementation, tokenizer, and bundled
`adaptcliplib/bpe_simple_vocab_16e6.txt.gz` vocabulary derive from OpenAI CLIP,
through AdaptCLIP. The complete original MIT notice is included in this release.
The backbone weights are downloaded separately and are not included in this
source repository.

## OpenCLIP

- Source: <https://github.com/mlfoundations/open_clip>
- License: [MIT license and copyright notice](licenses/OpenCLIP-MIT.txt)

Model loading and checkpoint conversion utilities, preprocessing helpers, and
associated constants in `adaptcliplib` contain OpenCLIP-derived code. The full
MIT notice, including the original author copyright list, is preserved in the
linked license file. Separate model downloads do not become part of this source
release.

## Public loss and evaluation implementations

The research code inherited metric implementations carrying Intel/Anomalib
Apache-2.0 notices and a focal-loss implementation attributed to
[Loss_ToolBox-PyTorch](https://github.com/Hsuxu/Loss_ToolBox-PyTorch), whose
upstream repository is licensed under Apache-2.0. These inherited implementations
are not distributed in this release.

`adaptcliplib/loss.py::FocalLoss` was newly implemented for this release from the
focal-loss formula using indexed probability selection. The focal-loss method is
described in Tsung-Yi Lin et al., *Focal Loss for Dense Object Detection* (2017),
<https://arxiv.org/abs/1708.02002>. Its public API and the research configuration's
probability-smoothing convention are retained. Numerical tests cover analytical
loss values and derivatives, class weights, spatial ordering, and composition
with softmax. The remaining helpers in `adaptcliplib/loss.py` are retained from
the AdaptCLIP-based research code.

The public evaluation utilities implement the required metric interfaces without
bundling the inherited Anomalib `metrics/` directory. The project documentation
records the supported metrics and validation scope; this provenance note does
not assert equivalence of every historical evaluation option.

## External dependencies and data

Python packages installed through the environment specification retain their own
licenses. Their source code and distributions are not vendored here. Datasets,
research logs, trained checkpoints, and paper PDFs are not part of this source
release. Obtain external assets from their respective providers and follow their
terms.
