#!/usr/bin/env bash
# A whole directory of CIFs through the GPoreU command-line interface.
#
# Every CIF below the given path(s) is characterized independently into
# out_directory/NAME.summary.  A structure that fails is reported but
# does not stop the batch.  Files and directories may be mixed.
#
# Usage:  ./run_directory.sh [DIR_or_file ...]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -gt 0 ]; then
    INPUTS=("$@")
else
    INPUTS=("$HERE/cifs")      # ships with RON, STO and WEI
fi

gporeu "${INPUTS[@]}" -o out_directory

echo "--> one summary per structure in out_directory/"
