# AILIS EMBER-Harness integration

`ailis-ember-harness.cjs` is the actual standalone stage-gate module copied from
`electron/ailis-ember-harness.cjs` in the AILIS integration checkout at commit
`f17c4b67210a94dbf1dac945a8bd56e3cd484c92` (branch `FEAT/ember-harness`).
It depends only on Node.js `crypto`; the rest of the desktop application is not
duplicated here. Its original MIT notice is retained.

- [Public AILIS source](https://github.com/haowenGuo/AILIS)
- [Anonymous implementation mirror](https://anonymous.4open.science/r/AILIS-4ABB)

The snapshot is for reproducing the integration used during the EMBER research,
not a claim that it matches the current AILIS desktop release. Use the AILIS
repository for the actively maintained desktop application.
