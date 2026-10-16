# 04 · A directory of CIFs, command line

Point `gporeu` at a folder and every structure in it is characterized
independently.

## Run

```bash
./run_directory.sh                       # uses the bundled cifs/ (3 CIFs)
./run_directory.sh /path/to/cifs         # or any directory
./run_directory.sh dirA dirB one more.cif   # mix freely
```

## What happens

- Each `.cif` found under the input paths runs the default full
  characterization independently, one GPU pipeline pass per structure.
- Reports: `out_directory/NAME.summary` for every structure.
- A structure that cannot be parsed or processed is reported as failed
  and the batch moves on to the next file; the exit code is `1` if any
  failed, `0` if all succeeded.

## Sample output (tail)

```
====================================================================
 GPoreU characterization summary - WEI.cif
 ...
====================================================================
 processed 3/3 structures -> out_directory
--> one summary per structure in out_directory/
```
