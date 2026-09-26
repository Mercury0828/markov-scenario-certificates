# Scenario certificates from a Markov trajectory: code and data

Code and data that reproduce every computed figure, table and number of the paper

> Jiachen Shen and Hui Zhong, *Scenario optimization with Markovian data: a posteriori risk certificates from a single
> trajectory*, manuscript, 2026.

The paper gives an a posteriori risk certificate for scenario decisions computed from one trajectory of a stationary,
uniformly mixing Markov chain, and shows that its dependence on the mixing time cannot be avoided. Each script here writes
its output to `results/`, and `scripts/check_paper.py` compares the regenerated values with the values printed in the
paper.

## Quick start

```
pip install -r requirements.txt
python run_all.py --use-data      # about 12 minutes: builds Table 2 from the archived runs in data/reserve
python run_all.py --jobs 8        # about 30 minutes: also reruns the Monte Carlo study (2.5 hours on one core)
```

`run_all.py` runs the scripts below in order and ends with the comparison, which prints one `PASS`/`FAIL` line per
printed value (81 checks).

## What reproduces what

| Paper | Script | Output in `results/` | Kind | Time (1 core) |
|---|---|---|---|---|
| Example 2.6; worked computation after Theorem 3.1; limits after Proposition 4.6; Remark 4.7; reserve model facts (§5.2) | `scripts/text_numbers.py` | `text_numbers.json` | exact | seconds |
| Figure 1 | `scripts/figure1_sticky_path.py` | `figure1_sticky_path.pdf/.png/.json` | two sample paths, fixed seeds | seconds |
| Figure 3 and the numbers of §5.1 | `scripts/figure3_certificate.py` | `figure3_certificate.pdf/.png/.json` | exact | about 10 minutes |
| Table 3 and the numbers of §5.3 | `scripts/table3_validity.py` | `table3_validity.tex/.json` | exact | seconds |
| Monte Carlo runs for Table 2 | `scripts/reserve_simulate.py` | `reserve/*.npz` | Monte Carlo, fixed seeds | about 2.5 hours |
| Table 2 and the numbers of §5.2 | `scripts/table2_reserve.py` | `table2_reserve.tex/.json` | from the runs | seconds |
| comparison with the paper | `scripts/check_paper.py` | printed | — | seconds |

Figure 2 is a schematic drawn in the LaTeX source of the paper, and Table 1 compares results from the literature; neither
contains computed numbers. `expected/` holds Tables 2 and 3 as printed in the paper.

## Library

`mixscen/` holds the formulas; the scripts only tabulate and plot.

- `certificates.py`: the certificate ε_UB(k) of Theorem 3.1; the compression certificate for independent data with the
  weights w_k = 1/((k+1)(k+2)) of Remark 3.5 and with equal weights w_k = 1/n; the Beta plug-in 1 − β^{1/N}; the
  thinning baseline of §5.2 (retained sample size, confidence level, gap chosen before the data).
- `sticky.py`: the sticky chain of Section 4 (refresh probability (4.1), mixing time, risk quantile of the largest value)
  and the threshold ε♯ of Theorem 4.3(a), computed by root finding (Brent's method) on binomial distribution functions.
- `chains.py`: the chains of §5.3 and the exact exceedance probabilities of Table 3. For a finite chain, the probability
  that n retained states stay in a set S is π_S Q_S^{n−1} 1 with Q_S the kernel restricted to S.
- `reserve.py`: the reserve problem of §5.2 (demand model, greedy reconstruction, exact risk V(x), simulation).

## Data

`data/reserve/` contains the Monte Carlo runs of Table 2 used in the paper: for each setting (`t5_N1e4`, `t5_N1e5`,
`t20_N1e4`, `t20_N1e5`, the rows of Table 2 from top to bottom) 20 files `<setting>_chunk<c>.npz` of 5000 trajectories
each. Every file holds one value per trajectory: `k`, `V` and `cost` (observed complexity, exact risk and cost of the
decision computed from all N scenarios) and `kthin_<beta>`, `Vthin_<beta>`, `costthin_<beta>` (the same for the
thinned sample used at confidence level beta = 0.001 or 0.05). `scripts/reserve_simulate.py` regenerates these files bit
for bit.

## Random numbers

Only Figure 1 and Table 2 use random numbers, both through NumPy's default generator (PCG64,
`numpy.random.default_rng`).

- Figure 1: panel (a) uses seed 0; panel (b) uses the first seed s = 0, 1, 2, … whose path has round(1 + 59q) fresh values
  (seed 13).
- Table 2: each of the four settings (rows of the table, s = 0, …, 3 from top to bottom) is simulated in 20 chunks of 5000
  independent trajectories; chunk c of setting s uses seed 20260927 + 1000 s + c. Within a chunk, the generator first draws
  the initial regimes of the 5000 trajectories and then proceeds in blocks of 200 time steps. In each block it draws, in
  this order, the regime refresh indicators, the regimes of the refreshes, the common shocks and the regional noise, each
  as one array over (time, trajectory) and, for the noise, region. The first time step keeps the initial regime.
  The thinned samples are read off the same trajectories.

The exact probabilities (Figure 3, Table 3, the numbers in the text) do not depend on random numbers.

## Environment

The results in the paper were computed with Python 3.11.4, NumPy 2.2.6, SciPy 1.17.1 and Matplotlib 3.10.8 on Windows 11
(`requirements.txt`). With these versions the scripts reproduce the paper's tables character for character and the
underlying values bit for bit. Other versions may change the last digits of floating-point results; the random streams
are fixed by NumPy's PCG64 and the draw order described above. The figures use the Computer Modern font that ships with
Matplotlib.

## Citation, licence and AI use

Please cite the paper and this repository (`CITATION.cff`). The code and data are released under the MIT licence
(`LICENSE`). Generative AI tools were used to write the code; the authors reviewed and verified it.
