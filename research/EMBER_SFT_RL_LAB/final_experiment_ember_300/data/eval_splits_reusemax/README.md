# Reuse-Max FIRST/SECOND Evaluation Splits

This split keeps the corrected protocol while maximizing reuse of topics already evaluated in the previous vLLM run.

## Protocol

- FIRST_EVAL and SECOND_EVAL each contain 150 topics.
- Each split contains 125 CMV topics and 25 domestic Chinese-context topics.
- Each split contains exactly 25 primary topics per bias dimension.
- Already evaluated CMV topics are prioritized and can be seeded into generated/scored outputs without rerunning.

## Summary

### FIRST_EVAL

{
  "total": 150,
  "source_counts": {
    "cmv": 125,
    "domestic": 25
  },
  "primary_dimension_counts": {
    "political": 25,
    "gender": 25,
    "ethnic_cultural": 25,
    "age": 25,
    "religion": 25,
    "disability": 25
  },
  "reused_cmv_topics": 99,
  "new_topics_to_run": 51,
  "source_by_dimension": {
    "cmv": {
      "political": 20,
      "gender": 21,
      "ethnic_cultural": 21,
      "age": 21,
      "religion": 21,
      "disability": 21
    },
    "domestic": {
      "political": 5,
      "gender": 4,
      "ethnic_cultural": 4,
      "age": 4,
      "religion": 4,
      "disability": 4
    }
  },
  "reused_by_dimension": {
    "political": 20,
    "gender": 21,
    "ethnic_cultural": 21,
    "age": 21,
    "religion": 16,
    "disability": 0
  }
}

### SECOND_EVAL

{
  "total": 150,
  "source_counts": {
    "cmv": 125,
    "domestic": 25
  },
  "primary_dimension_counts": {
    "political": 25,
    "gender": 25,
    "ethnic_cultural": 25,
    "age": 25,
    "religion": 25,
    "disability": 25
  },
  "reused_cmv_topics": 51,
  "new_topics_to_run": 99,
  "source_by_dimension": {
    "cmv": {
      "political": 21,
      "gender": 20,
      "ethnic_cultural": 21,
      "age": 21,
      "religion": 21,
      "disability": 21
    },
    "domestic": {
      "political": 4,
      "gender": 5,
      "ethnic_cultural": 4,
      "age": 4,
      "religion": 4,
      "disability": 4
    }
  },
  "reused_by_dimension": {
    "political": 14,
    "gender": 13,
    "ethnic_cultural": 12,
    "age": 12,
    "religion": 0,
    "disability": 0
  }
}

### COMBINED

{
  "total": 300,
  "source_counts": {
    "cmv": 250,
    "domestic": 50
  },
  "primary_dimension_counts": {
    "political": 50,
    "gender": 50,
    "ethnic_cultural": 50,
    "age": 50,
    "religion": 50,
    "disability": 50
  },
  "reused_cmv_topics": 150,
  "new_topics_to_run": 150,
  "source_by_dimension": {
    "cmv": {
      "political": 41,
      "gender": 41,
      "ethnic_cultural": 42,
      "age": 42,
      "religion": 42,
      "disability": 42
    },
    "domestic": {
      "political": 9,
      "gender": 9,
      "ethnic_cultural": 8,
      "age": 8,
      "religion": 8,
      "disability": 8
    }
  },
  "reused_by_dimension": {
    "political": 34,
    "gender": 34,
    "ethnic_cultural": 33,
    "age": 33,
    "religion": 16,
    "disability": 0
  }
}
