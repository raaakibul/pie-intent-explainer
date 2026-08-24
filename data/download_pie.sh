#!/usr/bin/env bash
# Downloads the PIE dataset annotation toolkit and prints instructions for
# obtaining the raw videos (which PIE's authors distribute on request).
#
# Usage: bash data/download_pie.sh /path/to/PIE_dataset
set -euo pipefail

TARGET_DIR="${1:-./PIE_dataset}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "==> Cloning PIE annotation/toolkit repo into $TARGET_DIR/PIE ..."
if [ ! -d "PIE" ]; then
  git clone https://github.com/aras62/PIE.git
else
  echo "PIE/ already exists, skipping clone."
fi

cat <<'EOF'

==> Next steps (manual, per PIE's license):

1. Request access to the raw PIE videos from the dataset authors:
   https://data.nvision2.eecs.yorku.ca/PIE_dataset/
   (the clone above only gives you pie_data.py + annotation XML/text files)

2. Once you have the videos, place them under:
     <TARGET_DIR>/PIE/PIE_clips/
   following the set/video naming convention documented in PIE/README.md.

3. Extract frames (PIE ships a helper for this):
     cd PIE && python clip_to_frames.py

4. Point configs/default.yaml -> data.pie_root at <TARGET_DIR>/PIE

EOF

echo "Done."