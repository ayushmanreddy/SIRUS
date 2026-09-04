# Epic 2: local model infrastructure

This increment adds a versioned local model manifest, checksum verification, path-traversal protection, safe CPU/GPU discovery, and `GET /system/preflight`.

1. Put approved weights in the untracked `models/` folder.
2. Copy `configs/model-manifest.example.json` to `models/manifest.json`.
3. Replace every example filename and checksum with the locally approved values.
4. Use `APP_PROFILE=offline-demo`; public model, embedding, and vector URLs are blocked.
5. Call `/system/preflight`; artifacts are ready only if every checksum matches.

Model weights, credentials, source download URLs, prompts, and confidential documents stay out of Git.
