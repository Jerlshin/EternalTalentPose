# artifacts/ — the offline→online contract (committed doc only)

This directory is **gitignored**. Its contents are *derived fact*: built once
by the offline pipeline, hash-pinned in `MANIFEST.json`, and consumed read-only
by the online pipeline. Artifacts are never hand-edited.

## Rebuild

```bash
make build
# equivalently:
uv run redstack build --config configs/runtime/offline.yaml
```

This regenerates every artifact (O0–O18) and writes a fresh `MANIFEST.json`
whose per-key sha256 the online `ArtifactStorePort` verifies at R0. Any hash
mismatch aborts the run — there is no degraded mode.

Only this `README.md` is committed; all produced artifacts stay local.
