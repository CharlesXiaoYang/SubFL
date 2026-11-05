# SubFL

Code and results for the paper *SubFL: Public-Subspace Gradient Perturbation for Locally Differentially Private Federated Learning* (Yang Xiao, Wenbo He).

In SubFL every client trains with record-level differential privacy, so no server has to be trusted. At every local step, each private per-example gradient is split into its part inside a subspace estimated from public data and the remainder. The two parts are clipped and noised separately, and their privacy cost is accounted jointly with Rényi DP (δ = 10⁻⁵). We compare SubFL with full-dimension client-side DP-SGD ("dense") on CIFAR-10 (ResNet-20, ResNet-56) and SVHN (ResNet-20), and with PrivateFL on CIFAR-10.

## Repository layout

| Path | What it is |
|---|---|
| `train.py` | Federated training. `--arm gep_step_keep` is SubFL, `--arm dense` is client-side DP-SGD, `--arm nonprivate` trains without clipping or noise. |
| `train_privatefl.py` | The PrivateFL baseline: client-side DP-SGD with PrivateFL's input or feature transformation, or a frozen backbone with a DP linear head. |
| `pretrain.py` | Pretrains the warm-start ResNet-20 on public CIFAR-100. |
| `subfl/experiments/fl_local_gep_ldp.py` | The federated training loop of all arms: local DP steps, aggregation, evaluation, records. |
| `subfl/experiments/gep_step.py` | Per-example gradients, the public anchor subspace, and the noisy split of a lot's gradient. |
| `subfl/privacy/` | RDP accounting and noise calibration (`rdp.py`), Poisson sampling of local lots (`sampling.py`), per-tensor subspaces (`update_space.py`). |
| `subfl/data/` | CIFAR-10 and SVHN loaders, the public/private split and the IID or Dirichlet client partition. |
| `subfl/models/` | CIFAR ResNet-20 and ResNet-56 (GroupNorm). |
| `subfl/config/`, `subfl/analysis/`, `subfl/utils/`, `subfl/federated/` | Run configuration, run records (metrics, manifest), evaluation and small helpers. |
| `baselines/privatefl.py` | Our implementation of PrivateFL's input and feature transformations. |
| `experiments/*.txt` | The exact command of every run in the paper, one per line, grouped by study. |
| `run.sh` | Runs the commands of one `experiments/*.txt` file, optionally split across GPUs. |
| `checkpoints/c100_init.pt` | The CIFAR-100 ResNet-20 checkpoint used for all warm-started runs (61.8% CIFAR-100 test accuracy). |
| `results/runs/` | The records of all 263 runs in the paper, plus the pretraining run. |
| `results/tables/` | The tables built from those records by `make_tables.py`. |
| `results/figures/` | Fig. 2 (`main.pdf`) and Fig. 3 (`curves.pdf`), drawn by `make_figures.py`. |
| `make_tables.py`, `make_figures.py` | Build the tables and figures from the run records. |
| `Dockerfile`, `requirements.txt` | The environment we ran in (Python 3.11, PyTorch 2.7.0 with CUDA 12.8). |

Each run directory `results/runs/<study>/<run name>/<timestamp>_<experiment>/` holds:

- `metrics.csv`: test accuracy and training statistics at each evaluation (every other round and round 100);
- `manifest.json`: the full configuration;
- `run_metadata.json`: the privacy accounting (achieved ε, noise multipliers, client sizes) and, for SVHN and ResNet-56, time and memory;
- `run_status.json`: completion status and end time.

The pretraining run is at `results/runs/pretrain/<timestamp>_public_pretrain_cifar_resnet20_c100e100/`, with `metrics.csv` (per epoch) and `manifest.json`.

## Tables and figures from our runs

```bash
python3 make_tables.py     # Python >= 3.10, standard library only
python make_figures.py     # needs matplotlib 3.10.8
```

This rewrites `results/tables/` and `results/figures/` from `results/runs/`. A run's accuracy is the mean test accuracy over its last 10 evaluations (rounds 83 to 100). SubFL and the baselines are compared per seed with a paired t-test and a 95% confidence interval; the p-values of Tables IV, V and VI are Holm-adjusted.

| Output | In the paper |
|---|---|
| `main_cells.csv`, `main_tests.csv` | Table IV and Fig. 2 |
| `clips` column of `main_cells.csv`, `svhn_cells.csv`, `r56_cells.csv` | Table III |
| `ceilings.csv` | Non-private reference in Fig. 2 |
| `pfl_cells.csv`, `pfl_tests.csv` | Table V (`pfl_*`: each client's deployed model, averaged over clients; `pfl_global_*`: the global model alone) |
| `noniid_tests.csv` | Table VI |
| `controls.csv` | Table VII |
| `levers.csv` | Table VIII |
| `svhn_cells.csv`, `svhn_tests.csv`, `r56_cells.csv`, `r56_tests.csv` | Table IX |
| `curves.csv`, with `main_tests.csv` and `noniid_tests.csv` for the Δ labels | Fig. 3 |
| `runtime.csv`, `svhn_cost.csv`, `r56_cost.csv` | Training time and memory |
| `runs_used.csv` | The run record behind each ResNet-20 CIFAR-10 SubFL, dense and non-private result |

The clip norms in Table III were selected by validation on the public pool (`--val_source public_pool`, seed 4321); those selection runs are not included.

## Re-running the experiments

Build the image and download the data (CIFAR-100 for pretraining is downloaded by `pretrain.py`):

```bash
docker build -t subfl .
subfl() { docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/workspace subfl "$@"; }
subfl python -c "from torchvision.datasets import CIFAR10; CIFAR10('data/cifar10', download=True)"
subfl python -c "from torchvision.datasets import SVHN; [SVHN('data/svhn', split=s, download=True) for s in ('train', 'test', 'extra')]"
```

Without Docker, use Python 3.11, run `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128`, and drop the `subfl` prefix.

Run a study on one GPU, or split it across 4 GPUs:

```bash
subfl bash run.sh experiments/main.txt
for g in 0 1 2 3; do subfl bash run.sh experiments/main.txt $g 4 & done
```

| File | Runs | Paper |
|---|---|---|
| `experiments/main.txt` | 80 | Tables III, IV, V, VII, VIII; Figs. 2, 3; training time |
| `experiments/nonprivate.txt` | 6 | Non-private reference in Fig. 2 |
| `experiments/noniid.txt` | 40 | Tables V, VI; Fig. 3 |
| `experiments/controls.txt` | 15 | Table VII |
| `experiments/levers.txt` | 12 | Table VIII |
| `experiments/privatefl.txt` | 80 | Table V |
| `experiments/svhn.txt` | 20 | Tables III, IX; training time and memory |
| `experiments/resnet56.txt` | 10 | Tables III, IX; training time and memory |
| `experiments/pretrain.txt` | 1 | Warm-start checkpoint (optional) |

Records are written to `runs/<study>/`, logs to `logs/<study>/`, and completed runs are skipped when a file is run again. On one RTX 4090, a run takes about 4.2 hours with SubFL and 0.9 hours with dense DP-SGD on CIFAR-10 with ResNet-20 (SVHN: 6.6 and 1.4 hours; ResNet-56: 9.0 and 1.5 hours). Once all studies have finished, build the tables from the new runs with

```bash
python3 make_tables.py --runs runs --out new_tables
python make_figures.py --tables new_tables --out new_figures
```

To use a newly pretrained checkpoint instead of `checkpoints/c100_init.pt`, copy `runs/pretrain/<run dir>/best_checkpoint.pt` to `checkpoints/c100_init.pt` and replace the hash prefix `9e001bbeca181a08` in `experiments/*.txt` with the first 16 hex digits of its SHA-256.

### Run names

- `q1_` from scratch, `q2_` warm-started; CIFAR-10, ResNet-20, IID clients.
- Then `subfl` or `dense`, `e<ε>`, and `s<seed>`. Between `e<ε>` and `s<seed>`, a part such as `_Ce8_Cr4` (SubFL) or `_C8` (dense) gives the clip norms of a cell whose clips were selected at its own ε.
- `q5_<init>_a<α>_`: non-IID clients, Dirichlet concentration α.
- Controls: `q3_subfl_nores` (no residual), `q3_rand_res` and `q3_rand_nores` (random subspace), `rl_subfl_random` (random anchor labels), `q3_dense_c2` (dense with C = 2).
- `lev_const_` constant learning rate; `lev_r0.4_subfl` and `lev_r0.6_subfl` budget split between the two parts.
- `c_nonprivate_lr0.3_` and `cw_nonprivate_lr0.03_`: non-private reference, from scratch and warm-started, with their learning rates.
- `pf<ε>_<variant>_<init>[_a<α>]`: PrivateFL; `in` input transformation with the whole model trained, `fn` frozen backbone with feature transformation and a DP linear head, `lp` frozen backbone with a DP linear head; `_a<α>` as for `q5_`.
- `svhn_e<ε>_`, `r56_e16_`: SVHN with ResNet-20, CIFAR-10 with ResNet-56.
- `c100e100`: the CIFAR-100 pretraining run (100 epochs).
- Five seed-1234 runs are stored under their original names: `p_subfl_real` (`q1_subfl_e16_s1234`), `p_dense` (`q1_dense_e16_s1234`), `p_rand_res_real` (`q3_rand_res_s1234`), `p_rand_nores_real` (`q3_rand_nores_s1234`), `p_subfl_random` (`rl_subfl_random_s1234`).

## Acknowledgements

`baselines/privatefl.py` re-implements PrivateFL (Yang et al., USENIX Security 2023, https://github.com/BHui97/PrivateFL) in our training loop. The subspace projection builds on GEP (Yu et al., ICLR 2021). No third-party code is included.

## License

MIT (see `LICENSE`).
