---
name: transit-probe
description: Run the bundled transit API quality probe to audit OpenAI- or Anthropic-compatible relay endpoints for model degradation, prompt caching authenticity, and baseline similarity. Use when the user wants to test a transit base URL plus API key, export structured probe reports, build baseline profiles from historical runs, or compare an endpoint against known flagship or small-model samples.
---

# Transit Probe

## Overview

Use the bundled probe to run four degradation dimensions plus prompt caching checks against a transit endpoint. Prefer this skill when the task is operational API verification, not general model evaluation theory.

## Workflow

1. Decide how to supply credentials.
2. Run the probe with either direct CLI arguments or a config file.
3. Export a run report if the user wants repeatable records or later calibration.
4. Build or apply a baseline profile only when the user wants cross-run comparison.
5. Interpret results conservatively:
   - Treat degradation risk as heuristic unless a calibrated baseline profile exists.
   - Treat cache as real only when usage signals and TTFT improvement both support it.

## Quick Start

Run the wrapper script:

```bash
python3 scripts/transit_probe.py --base-url https://example.com/v1 --api-key sk-xxx
```

Use config-file mode when the user wants a reusable local setup:

1. Copy `assets/probe.ini.example` to a writable location such as the repo root or current workspace as `probe.ini`.
2. Fill `base_url`, `api_key`, optional `model`, and any report or baseline paths.
3. Run:

```bash
python3 scripts/transit_probe.py --config /absolute/path/to/probe.ini
```

## Common Operations

- Run the full default probe:
```bash
python3 scripts/transit_probe.py --config /absolute/path/to/probe.ini
```

- Run only degradation cases:
```bash
python3 scripts/transit_probe.py --config /absolute/path/to/probe.ini --skip-cache
```

- Export a structured report:
```bash
python3 scripts/transit_probe.py --config /absolute/path/to/probe.ini --report-out reports/latest.json
```

- Build a baseline profile from historical reports:
```bash
python3 scripts/transit_probe.py --build-baseline reports --baseline-out baselines/main.json --baseline-name main-profile
```

- Compare a live run against a baseline:
```bash
python3 scripts/transit_probe.py --config /absolute/path/to/probe.ini --baseline-profile baselines/main.json
```

## Result Interpretation

- Use `Atomic pass rate` and `Strict case pass rate` as operational quality signals.
- Treat `Risk` as a calibrated conclusion only if baseline profiles were built from representative flagship and small-model samples.
- For caching, require both:
  - a positive cache usage field such as `cached_tokens` or `cache_read_input_tokens`
  - a clear TTFT reduction on the second request
- If only the usage field changes but latency barely moves, report it as likely forged or ineffective caching.

## Resources (optional)

### scripts/
- `scripts/transit_probe.py`: Wrapper that resolves the repo root and executes the maintained probe via `uv run --project <repo_root> app.py`.

### assets/
- `assets/probe.ini.example`: Config template for reusable endpoint settings.
