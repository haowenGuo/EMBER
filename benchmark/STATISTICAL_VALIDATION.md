# Statistical Validation Notes

The file `results/sample_level_paired_statistics.csv` reports paired sample-level tests. A sample is defined by the same model, topic id, and debate round. For each mitigation method, the paired difference is:

`delta_i = score_BASE_i - score_METHOD_i`

Positive values indicate bias reduction relative to BASE. The table reports mean reduction, bootstrap 95% confidence intervals, a one-sided paired sign-flip randomization p-value, and Holm-adjusted p-values across all comparisons.

Human validation is reported as limited human sampling and three-source consistency evidence. The available local files do not contain multiple independent human annotators for every sampled item, so the package does not report Cohen's kappa or Krippendorff's alpha.
