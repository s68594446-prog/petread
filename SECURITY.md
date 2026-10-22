# Security

## Reporting a vulnerability

Use GitHub's **Security → Report a vulnerability** on this repository when available:
https://github.com/s68594446-prog/petread/security/advisories/new

If private reporting is unavailable, open a public issue asking for a private contact
channel **without including exploit details, credentials, or personal data**.
Include affected commit, reproduction steps using synthetic data, impact, and any
proposed fix in the private report. No response-time commitment is currently offered.

## Supported versions

Security fixes target the latest default branch. Older snapshots have no separate
support schedule; review updates before deploying.

## Deployment and data

Keep secrets in local environment variables or your deployment secret store.
Do not commit .env files or real user data. Rotate any exposed credential; deleting
it from the latest commit does not remove it from Git history or existing copies.
Use trusted local environments unless you have configured authentication, HTTPS,
access controls, retention, and backups appropriate to your deployment.
Provider-enabled operations can transmit content to the configured external service.
Tests and CI must use synthetic inputs and must not require production secrets.
