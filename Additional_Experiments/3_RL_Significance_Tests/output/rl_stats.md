# RL significance tests (re-run, N=10 runs per condition)

## Transfer reward by difficulty (mean_reward per run)

| comparison | difficulty | A mean±std | B mean±std | Welch p | MW p | Cliff's δ |
|---|---|---|---|---|---|---|
| tran_snn vs exp_snn | easy | 177.0±25.2 | 159.0±33.7 | 0.2171 | 0.3847 | +0.24 |
| tran_snn vs exp_snn | hard | 111.0±41.3 | 81.4±36.7 | 0.1259 | 0.1212 | +0.42 |
| tran_snn vs exp_snn | medium | 156.3±31.1 | 125.7±43.3 | 0.104 | 0.08897 | +0.46 |
| tran_snn vs exp_snn | very_hard | 90.3±35.6 | 59.2±27.0 | 0.05248 | 0.08897 | +0.46 |
| tran_snn vs diss_snn | easy | 177.0±25.2 | 176.0±21.2 | 0.9268 | 0.8501 | +0.06 |
| tran_snn vs diss_snn | hard | 111.0±41.3 | 106.0±45.7 | 0.8105 | 0.9097 | +0.04 |
| tran_snn vs diss_snn | medium | 156.3±31.1 | 144.5±37.3 | 0.4773 | 0.4727 | +0.20 |
| tran_snn vs diss_snn | very_hard | 90.3±35.6 | 87.8±33.5 | 0.8784 | 0.9097 | +0.04 |
| tran_snn vs ann_raw | easy | 177.0±25.2 | 191.5±10.9 | 0.1389 | 0.08849 | -0.46 |
| tran_snn vs ann_raw | hard | 111.0±41.3 | 24.8±17.2 | 8.755e-05 | 0.0003298 | +0.96 |
| tran_snn vs ann_raw | medium | 156.3±31.1 | 85.8±55.2 | 0.004832 | 0.01133 | +0.68 |
| tran_snn vs ann_raw | very_hard | 90.3±35.6 | 20.2±8.3 | 0.0001864 | 0.0002461 | +0.98 |
| tran_snn vs ann_tran | easy | 177.0±25.2 | 178.8±29.2 | 0.8915 | 0.4725 | -0.20 |
| tran_snn vs ann_tran | hard | 111.0±41.3 | 106.8±48.8 | 0.8468 | 0.9698 | +0.02 |
| tran_snn vs ann_tran | medium | 156.3±31.1 | 155.4±37.9 | 0.9567 | 0.6776 | -0.12 |
| tran_snn vs ann_tran | very_hard | 90.3±35.6 | 82.1±36.3 | 0.6308 | 0.6232 | +0.14 |
| ann_tran vs ann_raw | easy | 178.8±29.2 | 191.5±10.9 | 0.2445 | 0.3811 | -0.24 |
| ann_tran vs ann_raw | hard | 106.8±48.8 | 24.8±17.2 | 0.0005686 | 0.0007685 | +0.90 |
| ann_tran vs ann_raw | medium | 155.4±37.9 | 85.8±55.2 | 0.006706 | 0.01402 | +0.66 |
| ann_tran vs ann_raw | very_hard | 82.1±36.3 | 20.2±8.3 | 0.0005552 | 0.0003298 | +0.96 |

## Generalization gap & retention (per-run)

| comparison | metric | A mean±std | B mean±std | Welch p | MW p | Cliff's δ |
|---|---|---|---|---|---|---|
| tran_snn vs exp_snn | avg_gap | 57.81±23.94 | 70.21±24.60 | 0.2929 | 0.273 | -0.30 |
| tran_snn vs exp_snn | performance_retention | 50.21±16.93 | 37.26±15.92 | 0.1119 | 0.07566 | +0.48 |
| tran_snn vs diss_snn | avg_gap | 57.81±23.94 | 63.21±27.59 | 0.6626 | 0.8501 | -0.06 |
| tran_snn vs diss_snn | performance_retention | 50.21±16.93 | 49.72±16.45 | 0.9516 | 0.8501 | -0.06 |
| tran_snn vs ann_raw | avg_gap | 57.81±23.94 | 147.91±29.17 | 1.415e-06 | 0.0003298 | -0.96 |
| tran_snn vs ann_raw | performance_retention | 50.21±16.93 | 10.62±4.46 | 4.289e-05 | 0.0001827 | +1.00 |
| tran_snn vs ann_tran | avg_gap | 57.81±23.94 | 64.04±28.38 | 0.6208 | 0.7337 | -0.10 |
| tran_snn vs ann_tran | performance_retention | 50.21±16.93 | 45.60±16.92 | 0.571 | 0.7337 | +0.10 |
| ann_tran vs ann_raw | avg_gap | 64.04±28.38 | 147.91±29.17 | 7.804e-06 | 0.0004396 | -0.94 |
| ann_tran vs ann_raw | performance_retention | 45.60±16.92 | 10.62±4.46 | 0.00012 | 0.0002461 | +0.98 |

## Omnibus test across SNN encoding groups

- easy: Kruskal-Wallis H=1.14, p=0.5661 (exp_snn, diss_snn, tran_snn)
- hard: Kruskal-Wallis H=2.11, p=0.3485 (exp_snn, diss_snn, tran_snn)
- medium: Kruskal-Wallis H=2.81, p=0.2453 (exp_snn, diss_snn, tran_snn)
- very_hard: Kruskal-Wallis H=4.28, p=0.1176 (exp_snn, diss_snn, tran_snn)
