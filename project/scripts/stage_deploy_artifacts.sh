#!/usr/bin/env bash
# Stage model outputs + zone CSV + labeled data so you can commit them for hosting.
# .gitignore normally excludes these paths; git add -f overrides that.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

git add -f data/aux/taxi_zone_centroids.csv
git add -f data/processed/labeled.parquet
git add -f outputs/

echo "Staged deploy artifacts under $ROOT."
echo "Next: git status && git commit -m 'Add artifacts for public deploy' && git push"
