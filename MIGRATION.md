# MiMo-V2.6-Flash migration (2026-10-10)

Review-only change based on main `42dabdf9192fee66261cb926684856519fecdf0a`.
No production deployment, production database access, or real API validation was performed.
Railway's effective config/environment and auto-deploy settings remain unverified.

## Minimal changes

- Active model ID becomes `mimo-v2.6-flash`; active documentation references updated.
- Keep prompts, knowledge, thinking enabled, fps=2, default resolution, confidence floor=0.5,
  retries, database schema and feedback handlers unchanged.
- Explicit `max_completion_tokens: 32768` retains the old default total output budget,
  including reasoning. It avoids the new 131072 default, without assuming a smaller
  thinking budget preserves quality. This is a per-call ceiling, not expected usage or
  a whole-job cost limit: frames mode can make three calls, each with retries.
- `media_format: mimo` selects MiMo wire format; generic OpenAI audio keeps its previous format.
- Non-stop completions and empty final content fail closed without retry. Reasoning
  content is never parsed as the final answer; stale raw text is cleared before each call.

## Verified official contract, not a live server test

| Input/option | Request shape |
| --- | --- |
| Text | Existing string / `type: text` parts |
| Image | `type: image_url`, `image_url.url`: public URL or MIME data URI |
| Video | `type: video_url`, `video_url.url`: public URL or MIME data URI |
| Video options | `fps` and `media_resolution` are siblings of `video_url` in the content part, never top-level or nested inside `video_url` |
| Audio | `type: input_audio`, `input_audio.data`: MIME data URI or public URL; no `format` field |
| Thinking | Top-level `thinking: {type: enabled}` (disabled is also supported) |
| JSON | Keep `response_format: {type: json_object}` and existing parser/filter |

MiMo fps range is 0.1–10, resolution `default` or `max`. No custom temperature/top_p
are sent; official thinking-mode values are forced by the server.
Audio MIME mapping is explicit for WAV/MP3/FLAC/M4A/OGG. Previously accepted upload
extensions AAC/OPUS/CAF/AMR now fail explicitly in the MiMo adapter; they need conversion
before inference. Do not claim these or HEIC/HEIF/M4V/MKV/WEBM/3GP are newly verified.
Image/video MIME fallbacks predate this change and remain a release risk for those
extensions; test actual uploaded media or convert them in a separately reviewed change.
Base64 strings have a documented 50 MB ceiling; current web upload limits (video 20 MiB,
image 10 MiB, audio 15 MiB) fit after encoding. CLI/public URL paths still need size checks.

## Offline regression suite

```sh
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

Tests stub HTTP and forbid socket connections. Synthetic media bytes only validate
serialization, not decoding. Tests exercise text/photo/audio/native/frames orchestration,
JSON parsing, threshold/refusal behavior, token usage, truncated-output rejection,
generic OpenAI audio compatibility, temporary SQLite record/export isolation and the
feedback handler. They do not measure model accuracy, actual codec support, FFmpeg
decoding, browser/SSE transport, latency or API authentication.
No real credentials were requested, retrieved or used. There was no safe project-specific
test credential provided to this task, so no real API calls were attempted.

Local result: 16 test methods passed on Python 3.14 in a temporary environment with
existing system dependencies and the declared `python-multipart` dependency installed.
Existing SQLite connection ResourceWarnings and FastAPI startup deprecation warnings
were observed; their underlying code is unchanged. Re-run on Railway's Python version
before release rather than treating this environment as production parity.

## Safe release gates (require separate deployment approval)

1. Inspect Railway's deployed commit, config path and environment overrides; confirm
   branch/PR creation does not deploy this branch. Leave production untouched for review.
2. Run the suite in the deployed Python/dependency environment. Back up persistent
   SQLite/uploads and record a known-good deploy artifact; no schema migration is needed.
3. In isolated staging, with credentials supplied through the existing secret manager,
   test consented samples of actual text, image, video-with-audio, audio, silence,
   missing/weak evidence and all production formats. Do not copy keys to logs or Git.
4. Evaluate a fixed labeled corpus against previously saved outputs: blind review
   candidate correctness, confidence/refusal, hallucinated signals and unsafe advice.
   JSON contract tests alone cannot guarantee unchanged judgment quality.
5. Check final `finish_reason=stop`, JSON fields, reasoning+output usage, cost and latency;
   verify stored model/usage, feedback, history/export and SSE end-to-end. If budget
   truncates, fail the sample and investigate before changing the cap; do not auto-increase.
6. Obtain approval before merging to an auto-deployed branch or deploying. Canary and
   monitor errors, refusal rate and costs. After old-ID retirement, rollback must keep a
   supported model/config; merely restoring an old `mimo-v2.5` deployment is not safe.

Official notice lists both 10:00 and 18:00 for Oct 14 routing; plan before **Oct 14
10:00 UTC+8**. Old ID stops working **Oct 21 10:00 UTC+8**.

## Official sources

- [Deprecation and default budget](https://mimo.mi.com/docs/en-US/updates/deprecate)
- [Chat request/response fields](https://mimo.mi.com/docs/en-US/api/chat/openai-api)
- [Video](https://mimo.mi.com/docs/en-US/quick-start/usage-guide/multimodal-understanding/video-understanding)
- [Images](https://mimo.mi.com/docs/en-US/quick-start/usage-guide/multimodal-understanding/image-understanding)
- [Audio](https://mimo.mi.com/docs/en-US/quick-start/usage-guide/multimodal-understanding/audio-understanding)
- [Thinking hyperparameters](https://mimo.mi.com/docs/en-US/api/guidance/model-hyperparameters)
- [Pricing](https://mimo.mi.com/docs/en-US/price/pay-as-you-go)
