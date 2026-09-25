# FIRST/SECOND Evaluation Split Protocol

This file documents the corrected split protocol after discovering that the old first-150 slice was ordered as CMV-only.

## Protocol

- FIRST_EVAL: 150 topics for first evaluation and downstream SFT/RL data construction.
- SECOND_EVAL: 150 disjoint topics for held-out testing of BASE, EMBER-PROMPT, EMBER-AGENT, SFT, and SFT+RL.
- Each split contains 125 CMV topics and 25 domestic Chinese-context topics.
- Each split is balanced by six primary bias dimensions: 25 topics per dimension.
- Domestic topics are allocated as evenly as possible because 25 is not divisible by 6.

## Why This Correction Was Needed

The earlier `final_ember_topics_300.jsonl` was globally balanced, but ordered as 200 CMV rows followed by 100 domestic rows. Running rows 1-150 therefore used only CMV topics and missed all domestic topics. It also missed disability as a primary topic because disability CMV topics were placed in rows 168-200.

## Split Summary

### FIRST_EVAL

- Total: 150
- Source counts: {'cmv': 125, 'domestic': 25}
- Primary dimension counts: {'political': 25, 'gender': 25, 'ethnic_cultural': 25, 'age': 25, 'religion': 25, 'disability': 25}
- Source by dimension: {'cmv': {'political': 20, 'gender': 21, 'ethnic_cultural': 21, 'age': 21, 'religion': 21, 'disability': 21}, 'domestic': {'political': 5, 'gender': 4, 'ethnic_cultural': 4, 'age': 4, 'religion': 4, 'disability': 4}}

### SECOND_EVAL

- Total: 150
- Source counts: {'cmv': 125, 'domestic': 25}
- Primary dimension counts: {'political': 25, 'gender': 25, 'ethnic_cultural': 25, 'age': 25, 'religion': 25, 'disability': 25}
- Source by dimension: {'cmv': {'political': 21, 'gender': 20, 'ethnic_cultural': 21, 'age': 21, 'religion': 21, 'disability': 21}, 'domestic': {'political': 4, 'gender': 5, 'ethnic_cultural': 4, 'age': 4, 'religion': 4, 'disability': 4}}
