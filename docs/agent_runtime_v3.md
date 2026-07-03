# Historical Agent Runtime V3 Notes

`agent_runtime_v3.md` is kept only as a historical compatibility path.

The current main-chain document is [agent_runtime.md](agent_runtime.md). New runtime design, debugging, tests, and development should use:

- `src/mahjong_agent_runtime`
- `scripts/run_agent_app.py`
- `scripts/agent_runtime_app.py`
- `scripts/run_agent_runtime_eval.py`
- `scripts/verify_agent_runtime_boundary.py`
- `tests/test_agent_runtime.py`

Do not continue main-chain development in historical v2/v3 entrypoints.
