# LCS Benchmark Suite

This folder contains a benchmark script to compare:

- **Suffix Automaton**
- **Enhanced Suffix Array** (Suffix Array + LCP)

for the **Longest Common Substring (LCS)** problem.

## What it tests

For multiple string lengths, the script generates reproducible scenarios:

- random uniform strings
- mutated motif implanted into larger host strings
- repetitive strings with noise
- near-identical strings with small mutation
- disjoint alphabets

The same generated string pair is always benchmarked with both algorithms.

## Run

First, create a venv or conda environment and install dependencies:

```bash
pip install -r Code/requirements.txt
```

Then, from the workspace root:

```bash
python Code/benchmark_lcs.py --lengths 100,250,500,1000,2000 --cases-per-length 5 --runs 100 --seed 42
```

If you wish to plot already existing results, you can run:

```bash
python Code/benchmark_lcs.py --mode plot
```

## Output

By default, outputs are written to `Code/results/`:

- `raw_runs.csv` (all single runs)
- `case_lcs_values.csv` (each generated string pair `s`, `t`, plus all LCS value(s))
- `summary_stats.csv` (mean, median, std, IQR, min, max)
- `tables/summary_<scenario>.tex` (LaTeX tables)
- `plots/<scenario>_<stats>.png` (runtime + memory graphs)
- `benchmark_config.txt` (used settings)

## Notes

- Runtime is measured with `time.perf_counter_ns`.
- Peak memory is measured with `tracemalloc`.
- A correctness check ensures both algorithms return the same LCS value(s).
