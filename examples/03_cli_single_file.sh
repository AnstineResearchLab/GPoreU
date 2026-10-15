#!/usr/bin/env bash
# One CIF through the command-line interface, full characterization.
#
# Writes out_single/NAME.summary (identical text to what is printed).
#
# Usage:  ./03_cli_single_file.sh [structure.cif]
set -euo pipefail

CIF="${1:-${GPOREU_TEST_CIF_DIR:-./cif_files}/IRMOF1.cif}"

gporeu "$CIF" -o out_single

echo "--> report saved to out_single/$(basename "${CIF%.cif}").summary"
