#!/usr/bin/env bash
# Turning specific calculations ON or OFF.
#
# GPoreU works with a set of user-facing *methods*; each method pulls in
# only the pipeline steps it actually needs (and nothing else).  Two
# equivalent ways to control which calculations run:
#
#   1. -m / --methods   comma-separated list on the command line
#   2. a JSON settings file (-c), see no_energy_network_sa.json here
#
# `--explain` previews the resolved plan; `--list-methods` documents
# every method, its aliases and what it requires.
#
# Usage:  ./turn_off_calcs.sh [structure.cif]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CIF="${1:-$HERE/cifs/IRMOF1.cif}"

echo '=== full default characterization (everything on) ==='
gporeu "$CIF" --explain

echo
echo '=== only PLD: PSD, surface area, pore network, energy all OFF ==='
gporeu "$CIF" -m pld --explain

echo
echo '=== keep the energy histogram but drop the surface area ==='
gporeu "$CIF" -m porosity,percolation,ffv,pore_size_distribution,pore_network,pld,energy_histogram --explain

echo
echo '=== the same choice expressed in a settings file ==='
gporeu "$CIF" -c "$HERE/no_energy_network_sa.json" --explain

echo
echo '=== command-line flags override the settings file ==='
gporeu "$CIF" -c "$HERE/no_energy_network_sa.json" -m lcd --explain

echo
echo '=== now actually run the leanest useful combination ==='
gporeu "$CIF" -m porosity,lcd -o out_lean

echo
echo 'Hint: drop --explain to run any plan above for real.'
