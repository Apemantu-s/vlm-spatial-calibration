# Fitting in 8 GB

50 optimizer steps each (`--limit 800 --epochs 1`), batch 2, accumulation 8,
images at most 448 px on the long side.

| | peak VRAM (torch) | nvidia-smi | samples / s |
|---|---|---|---|
| with gradient checkpointing | 4.54 GB | 4561 MiB | 5.59 |
| without | 5.74 GB | 5829 MiB | 9.1 |

Card: RTX5070 (Laptop)