# Figure scripts

Set `PANDA_DATA_ROOT` to the pipeline work folder (or pass it as the second argument), then:

```bash
python make_figures_1_2_4_7.py <output_dir>   # Figs 1, 2, 4, 7
python make_figures_14_15.py  <output_dir>    # Figs 14, 15
python make_S1_table.py       <output_dir>    # S1 Table
```

Each figure is written as PNG and PLOS TIFF (RGB, LZW, 300 dpi, 7.5 in wide). `provenance/` lists every source file, relative to `PANDA_DATA_ROOT`, with its SHA-256.

Figs 1, 2, 4 and 7 use PANDA training WSI `362d30ace92ce17128d176137362b8c2`. Vahadane normalization uses TIAToolbox dictionary learning, so the normalized patch can vary slightly between runs.
