# Legacy experiment scripts

This directory keeps the early EMBER experiment scripts used during thesis
development. They are preserved for traceability, but the cleaned public
reference implementation lives in `src/ember`.

Recommended public entry points:

- `src/ember/arena.py`: multi-agent adversarial dialogue arena
- `src/ember/agent.py`: EMBER-Agent reflection and revision loop
- `src/ember/harness.py`: EMBER-Harness stage gates, snapshots, and rollback
- `examples/`: minimal runnable demos without private API keys

Before running legacy scripts, create a local `.env` or edit a private copy of
the configuration. Do not commit API keys, local model paths, or raw outputs.
