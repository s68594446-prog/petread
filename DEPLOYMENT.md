# PetRead Free deployment

Release: `20261107-visual-v2.1-free.1`.

This release contains the yellow UI, official OpenAI adapter, durable device-scoped jobs,
and the `petread.visual.v2.1` evidence contract. Two independent reliable channels
including a main posture/ear axis are necessary, not sufficient, for image candidates.
Ambiguous evidence must still abstain. No confidence boosting or forced guesses.

Deployment settings:

- Existing Railway Free service and volume; do not delete or wipe the volume.
- Mount remains `/data`; new records use `DATA_DIR=/data/petread-v2` to leave the legacy database untouched.
- One worker and `PETREAD_MAX_ACTIVE_JOBS=1`.
- `OPENAI_API_KEY` is configured only in Railway Variables; never in the repository or image.
- `PETREAD_PUBLIC_ORIGIN=https://petread-production-bc55.up.railway.app` preserves
  same-origin checks and Secure cookies/HSTS behind TLS termination without trusting client proxy headers.
- Health probe: `/healthz`; readiness checks configuration, not paid model availability.
- Video limit: 10 MiB, 15 seconds, 1080p, 30fps. Free memory capacity is not a quality or load guarantee.

Deployment verification does not invoke models. OpenAI charges apply to later real
user analyses separately from Railway's free hosting allowance. Free instances may
sleep or exhaust their monthly allowance. This is a limited friend-testing release,
not an App Store readiness claim or an emotion-accuracy claim.

The release checkout preserves newer upstream commits. Local evaluation datasets,
raw results, credentials, and unrelated uncommitted assets are not published.
