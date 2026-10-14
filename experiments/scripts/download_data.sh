#!/bin/bash
# Download the four public PhysioNet datasets used in the study (run on a login node with internet).
#   bash scripts/download_data.sh /path/to/data
# Afterwards:  export DATA_ROOT=/path/to/data/physionet.org/files
#
# Sizes: PTB-XL 100 Hz part ~0.6 GB, PTB Diagnostic ~1.7 GB, STAFF III ~3.2 GB, NSTDB ~0.03 GB.
#
# Alternative (data already on Google Drive): copy the same folder layout with rclone, e.g.
#   rclone copy gdrive:ECG_data/physionet.org/files "$DATA_ROOT" --progress
# The loaders expect the original PhysioNet file structure (ptbxl_database.csv, records100/,
# RECORDS + patientXXX/ folders, staffiii data/ + spreadsheet, nstdb bw/em/ma records).
set -euo pipefail
DEST="${1:-$HOME/data}"
mkdir -p "$DEST"
cd "$DEST"
W="wget -r -N -c -np -nv"
$W --reject-regex '.*/records500/.*' https://physionet.org/files/ptb-xl/1.0.3/
$W https://physionet.org/files/ptbdb/1.0.0/
$W https://physionet.org/files/nstdb/1.0.0/
$W https://physionet.org/files/staffiii/1.0.0/
echo
echo "Done. Use:  export DATA_ROOT=$DEST/physionet.org/files"
