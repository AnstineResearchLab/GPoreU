#!/usr/bin/env bash
# A whole directory of CIFs through the command-line interface.
#
# Every CIF below the given path(s) is characterized independently into
# out_directory/NAME.summary; a structure that fails is reported but
# does not stop the batch.  Files and directories may be mixed.
#
# Usage:  ./04_cli_directory.sh [DIR ...]
set -euo pipefail

if [ "$#" -gt 0 ]; then
    INPUTS=("$@")
else
    INPUTS=("${GPOREU_TEST_CIF_DIR:-./cif_files}")
fi

gporeu "${INPUTS[@]}" -o out_directory

echo "--> one summary per structure in out_directory/"
