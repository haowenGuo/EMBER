# Data provenance and licensing

## Code license

Self-owned EMBER source code and accompanying software documentation are released
under the [MIT license](../LICENSE). The copied AILIS integration retains its
[original MIT notice](../integrations/ailis/LICENSE). No hosted service or
proprietary component is required to inspect this source or run the offline demos.

The MIT grant does **not** relicense third-party datasets, pretrained model
weights, quoted source passages, or generated responses containing those passages.

## Topic sources

The benchmark contains 300 topics: 200 sampled from Change My View (CMV) material
and 100 domestic-context supplementary topics. It records source identifiers,
construction methods, dimension labels and separate 150-topic FIRST/SECOND splits.

The historical CMV input is distributed upstream as
[`underscore2/changemyview_persuasion_kto`](https://huggingface.co/datasets/underscore2/changemyview_persuasion_kto).
Its dataset metadata at revision `e74d8c8bf4ed11f0e94788b1c88a72ba19bd96d2`
did not specify a license when checked on 2026-09-25. Public availability is not
a blanket copyright license. This release preserves provenance and makes no
additional license grant over those source texts. Check the upstream terms and
obtain any permission needed for your intended redistribution or use.

References to BBQ, CBBQ, BOLD and AgentDojo in experiment documentation identify
external research resources; they are not relicensed under this repository's MIT
license. Obtain their official versions and applicable terms from their original
maintainers. This release does not include pretrained base-model weights.

## Experimental outputs

Public data archives contain historical generated dialogues, evaluator outputs,
training-corpus exports and result tables. Preserve the original source and model
attribution when reusing them. Automatically generated risk labels are not human
ground truth. The files contain intentionally adversarial, biased or offensive
language used for safety evaluation; inclusion is not endorsement.

The formal-results archive also retains pilot runs, smoke checks, intermediate
review batches and failed-run metadata. A directory being included does not mean
its result is part of the final thesis comparison. Use the experiment protocol,
topic split, method label and run metadata before aggregating scores.

## Publication boundary

- API credentials are removed from public copies; environment files are examples.
- Original research files remain unchanged locally.
- Personal academic forms, reviewer correspondence, signatures, patent-application
  materials, private credentials, caches and bundled dependencies are excluded.
- Thesis full text is not included in this release pending a separate publication
  decision. Public result figures and research documentation remain available.
- No third-party model weights or server-only checkpoint files are represented as
  included. Re-running GPU experiments requires the applicable models and hardware.

The [release manifest](release-manifest.json) identifies the source-relative path,
published location, byte count and SHA-256 of each prepared research file.
