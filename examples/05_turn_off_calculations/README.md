# 05 · Turning calculations on and off

Control which characterizations run.  You select user-facing **methods**
(`gporeu --list-methods` for the table and aliases like `psd`, `sa`,
`ehist`); GPoreU adds exactly the pipeline steps those methods need —
nothing more.

## Run

```bash
./turn_off_calcs.sh            # plan previews (--explain) + one lean run
./turn_off_calcs.sh my.cif     # with another structure
```

equivalent to typing:

```bash
# everything (defaults):
gporeu cifs/IRMOF1.cif

# only the pore limiting diameter -- PSD, surface area, pore graph and
# the CH4 energy grids are all OFF:
gporeu cifs/IRMOF1.cif -m pld

# the same choice as a settings file (master switch = "methods" list):
gporeu cifs/IRMOF1.cif -c no_energy_network_sa.json

# flags override the file:
gporeu cifs/IRMOF1.cif -c no_energy_network_sa.json -m lcd
```

## What the previews show

```
=== full default characterization (everything on) ===
requested methods : porosity, percolation, ffv, pore_size_distribution,
                    surface_area, pore_network, pld, energy_histogram
execution plan    : grids -> connectivity -> psd -> surface_area -> network
                    -> pld -> energy_grid -> energy_hist

=== only PLD ===
execution plan    : grids -> connectivity -> pld

=== settings file (energy / network / surface area off) ===
execution plan    : grids -> connectivity -> psd -> pld
```

Note how dependencies flow: `energy_histogram` would normally drag in
the CH4 `energy_grid`; when it is not selected, no energy kernel runs at
all.  Aliases (`ehist` for `energy_histogram`, `sa`, `network`, ...) work
on the command line and in the JSON.

For the full documented set of options (grid spacing, seed, probes, PSD
cycles, pore-network buffers, energy cutoffs, output behavior) run
`gporeu --init-config my_settings.json` and pass it back with `-c`.
