#!/usr/bin/env bash
set -euo pipefail

if ! command -v detect-secrets >/dev/null 2>&1; then
  echo "detect-secrets not found. Install it with: pip install detect-secrets"
  exit 1
fi

echo "Generating secrets baseline (.secrets.baseline) - review before committing"
detect-secrets scan > .secrets.baseline

echo "Done. Commit the baseline if you wish to keep these results to suppress false positives."
