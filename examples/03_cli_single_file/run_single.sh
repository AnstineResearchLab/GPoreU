#!/usr/bin/env bash
# One CIF through the GPoreU command-line interface: full characterization.
#
# Writes out_single/NAME.summary (identical text to what is printed).
#
# Usage:  ./run_single.sh [structure.cif]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CIF="${1:-$HERE/cifs/IRMOF1.cif}"

gporeu "$CIF" -o out_single

echo "--> report saved to out_single/$(basename "${CIF%.cif}").summary"
