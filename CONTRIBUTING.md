# Contributing

Thank you for contributing. Start with the README and any repository AGENTS.md.
For larger changes, open an issue describing the problem and proposed scope.

## Development and verification

Use Python 3.13 in a virtual environment. Classroom Interpreter also requires Node.js 24.
Run the following from the repository root with the virtual environment activated:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use synthetic inputs and mocks. Passing tests does not establish real model quality,
hardware reliability, investment performance, or production acceptance.
Do not run paid provider calls, microphone capture, scheduled delivery, or data deletion
as part of an ordinary test run.

## Pull requests

- Keep changes focused and explain the user-visible problem and resulting behavior.
- Include relevant test commands, results, and remaining limitations.
- Add regression tests for behavior changes; avoid changing historical evidence to imply new validation.
- Use synthetic fixtures. Never include API keys, .env files, personal recordings,
  classroom transcripts, account data, or private machine configuration.
- Report security issues privately as described in [SECURITY.md](SECURITY.md).

Contributions are distributed under the repository's [MIT license](LICENSE).
Preserve third-party attribution and submit only material you have permission to contribute.
