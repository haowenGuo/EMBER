# FIRST_EVAL Reuse-Max Statistics Report

## Overall Variant Means

- llama BASE: mean=1.831, valid=886/900, std=1.886.
- llama EMBER-AGENT: mean=1.495, valid=883/900, std=1.422.
- llama EMBER-PROMPT: mean=1.497, valid=889/900, std=1.492.
- qwen BASE: mean=2.180, valid=880/900, std=1.781.
- qwen EMBER-AGENT: mean=1.639, valid=883/900, std=1.425.
- qwen EMBER-PROMPT: mean=1.858, valid=878/900, std=1.668.

## Pairwise Reduction vs BASE

- llama EMBER-AGENT: delta=-0.339, relative_reduction=18.472%, win=39.610%, worse=31.573%.
- qwen EMBER-AGENT: delta=-0.548, relative_reduction=25.159%, win=47.575%, worse=27.252%.
- llama EMBER-PROMPT: delta=-0.326, relative_reduction=17.891%, win=39.886%, worse=30.171%.
- qwen EMBER-PROMPT: delta=-0.316, relative_reduction=14.592%, win=43.372%, worse=28.721%.

## Files

- `overall_by_model_variant.csv`
- `round_curve.csv`
- `primary_dimension_overall.csv`
- `primary_dimension_round_curve.csv`
- `primary_dimension_round_curve_wide.csv`
- `bias_label_dimension_overall.csv`
- `bias_label_dimension_round_curve.csv`
- `bias_label_dimension_round_curve_wide.csv`
- `score_distribution_overall.csv`
- `score_distribution_by_round.csv`
- `score_bin_distribution_overall.csv`
- `score_bin_distribution_by_dimension.csv`
- `pairwise_by_model.csv`
- `pairwise_by_round.csv`
- `pairwise_by_primary_dimension.csv`
- `pairwise_by_source.csv`
- `topic_counts.csv`
