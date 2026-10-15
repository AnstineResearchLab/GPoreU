#!/usr/bin/env bash
# Turning specific calculations ON or OFF.
#
# GPoreU works with a set of user-facing *methods*; each method pulls in
# only the pipeline steps it actually needs (and nothing else).  Two
# equivalent ways to control which calculations run:
#
#   1. -m / --methods   comma-separated list on the command line
#   2. a JSON settings file (-c), see 05_turn_off_calcs_settings.json
#
# `--explain` previews the resolved plan; `--list-methods` documents
# every method, its aliases and what it requires.
#
# Usage:  ./05_cli_turn_off_calcs.sh [structure.cif]
set -euo pipefail

CIF="${1:-${GPOREU_TEST_CIF_DIR:-./cif_files}/IRMOF1.cif}"

echo '=== full default characterization (everything on) ==='
gporeu "$CIF" -o out_on --explain

echo
echo '=== only PLD: energy grids, PSD, pore network, SA all skipped ==='
gporeu "$CIF" -m pld -o out_pld_only --explain

echo
echo '=== keep the energy histogram but drop the surface area ==='
gporeu "$CIF" -m porosity,percolation,ffv,pore_size_distribution,pore_network,pld,energy_histogram -o out_no_sa --explain

echo
echo '=== the same choice expressed in a settings file ==='
gporeu "$CIF" -c "$(dirname "$0")/05_turn_off_calcs_settings.json" -o out_cfg --explain

echo
echo '=== and flags still override the settings file ==='
gporeu "$CIF" -c "$(dirname "$0")/05_turn_off_calcs_settings.json" \
    -m lcd --explain

echo
echo 'Hint: drop --explain to actually run; the plan above is what executes.'
