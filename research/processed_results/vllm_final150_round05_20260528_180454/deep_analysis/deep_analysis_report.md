# EMBER Prompt/Agent Effect Deep Analysis

## Key Findings

- final150_llama EMBER-PROMPT: mean 1.649 vs BASE 1.938, delta -0.290, relative reduction 14.936%.
- final150_llama EMBER-AGENT: mean 1.618 vs BASE 1.938, delta -0.320, relative reduction 16.503%.
- final150_qwen EMBER-PROMPT: mean 1.932 vs BASE 2.124, delta -0.192, relative reduction 9.044%.
- final150_qwen EMBER-AGENT: mean 1.777 vs BASE 2.124, delta -0.347, relative reduction 16.325%.

## Why the 3-topic test looked stronger


The final run is much broader: 150 topics, rounds 0-5, mixed source/dimension strata, and a local Qwen3.5-9B attacker. The earlier small runs only had 33 rows and were therefore very sensitive to topic selection and high-risk examples.

## Paired Bootstrap CI

- llama EMBER-AGENT: paired n=882, delta=-0.316, 95% CI [-0.443, -0.192], win=40.136%, worse=30.726%.
- qwen EMBER-AGENT: paired n=861, delta=-0.365, 95% CI [-0.488, -0.233], win=42.393%, worse=30.430%.
- llama EMBER-PROMPT: paired n=885, delta=-0.288, 95% CI [-0.417, -0.163], win=39.322%, worse=29.605%.
- qwen EMBER-PROMPT: paired n=860, delta=-0.177, 95% CI [-0.305, -0.030], win=40.349%, worse=30.581%.

## Baseline Ceiling/Floor Effect

- llama EMBER-AGENT base=0: n=225, delta=1.080, win=0.000%, worse=57.333%.
- llama EMBER-AGENT base=1: n=195, delta=0.395, win=31.795%, worse=42.051%.
- llama EMBER-AGENT base=3-4: n=193, delta=-1.223, win=68.394%, worse=6.218%.
- llama EMBER-AGENT base>=5: n=82, delta=-3.768, win=93.902%, worse=3.659%.
- llama EMBER-PROMPT base=0: n=223, delta=1.067, win=0.000%, worse=56.951%.
- llama EMBER-PROMPT base=1: n=196, delta=0.276, win=31.122%, worse=33.163%.
- llama EMBER-PROMPT base=3-4: n=194, delta=-1.134, win=70.619%, worse=11.340%.
- llama EMBER-PROMPT base>=5: n=81, delta=-3.519, win=90.123%, worse=3.704%.
- qwen EMBER-AGENT base=0: n=161, delta=1.037, win=0.000%, worse=60.248%.
- qwen EMBER-AGENT base=1: n=191, delta=0.445, win=25.131%, worse=41.885%.
- qwen EMBER-AGENT base=3-4: n=229, delta=-1.179, win=71.616%, worse=10.044%.
- qwen EMBER-AGENT base>=5: n=83, delta=-3.542, win=96.386%, worse=1.205%.
- qwen EMBER-PROMPT base=0: n=165, delta=1.291, win=0.000%, worse=56.364%.
- qwen EMBER-PROMPT base=1: n=190, delta=0.668, win=19.474%, worse=47.368%.
- qwen EMBER-PROMPT base=3-4: n=228, delta=-1.211, win=71.930%, worse=7.456%.
- qwen EMBER-PROMPT base>=5: n=79, delta=-3.190, win=91.139%, worse=3.797%.

## Generated Files

- `overall_summary.csv`
- `round_summary_deep.csv`
- `source_summary.csv`
- `primary_dimension_summary.csv`
- `pairwise_by_model.csv`
- `pairwise_by_round.csv`
- `pairwise_by_source.csv`
- `pairwise_by_primary_dimension.csv`
- `pairwise_by_base_bucket.csv`
- `dimension_label_summary.csv`
- `topic_variant_means.csv`
- `top_topic_improvements.csv`
- `top_topic_regressions.csv`
- `small_vs_final.csv`
