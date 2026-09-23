# TreeSBM data-split protocols (forecasting + clade holdout)

**Do not invent eval numbers.** These scripts only prepare data dirs; metrics come from later train/eval jobs.

## A) Temporal (flu primary; also COVID)

Goal: train on sequences/trees **before** a cutoff year, evaluate on later years (next season vs next few seasons).

| Dataset | Prep script | Default cutoffs | Output dir | Pipeline SLURM |
|---------|-------------|-----------------|------------|----------------|
| H3N2 HA | `scripts/prepare_h3n2_temporal.py` | train≤2022 / val=2023 / test=2024 | `data/h3n2` (or `data/h3n2_temporal_forecast`) | `slurm_h3n2_pipeline.sh` / `slurm_h3n2_temporal_forecast_pipeline.sh` |
| H1N1 HA | `scripts/prepare_h1n1_temporal.py` | train≤2023 / val=2024 / test=2025 | `data/h1n1_temporal` | `slurm_h1n1_temporal_pipeline.sh` |
| COVID Spike | `scripts/prepare_covid_temporal.py` | train≤2022 / val=2023 / test=2024–2025 | `data/covid_temporal` | `slurm_covid_temporal_pipeline.sh` |

Each prep writes `SPLIT_PROTOCOL.json` under the out-base (cutoffs, year histograms, group counts).

### How groups map to seasons

- **H3N2 / H1N1:** Groups are **date-ordered chunks** of `--group-size` sequences **within one year-band** (train / val / test). A group never crosses the year cutoff. Northern-hemisphere flu seasons roughly span Oct(Y−1)–Sep(Y); cutoffs here are **collection-year** cutoffs, not season labels. “Next season” ≈ `test_year = train_end + 2` with `val = train_end + 1`. “Next few seasons” widens `--test-end-year`.
- **COVID:** No flu season — calendar-year cutoffs. Groups remain **single-country**, date-ordered chunks (same tree shape as the geographic prep) but only using sequences inside that year-band.

### CLI examples

```bash
# H3N2 classic (reuse data/h3n2)
python scripts/prepare_h3n2_temporal.py

# H3N2 forecast dir with later cutoffs (needs 2025 source FASTA when available)
python scripts/prepare_h3n2_temporal.py \
  --out-base data/h3n2_temporal_forecast \
  --train-end-year 2023 --val-year 2024 \
  --test-start-year 2025 --test-end-year 2025

# H1N1 next-season
python scripts/prepare_h1n1_temporal.py
# multi-season test:
python scripts/prepare_h1n1_temporal.py --train-end-year 2022 --val-year 2023 \
  --test-start-year 2024 --test-end-year 2025

# COVID (requires spike extract first)
sbatch scripts/slurm_covid_extract.sh
python scripts/prepare_covid_temporal.py
python scripts/prepare_covid_temporal.py --train-end-year 2023 --no-val \
  --test-start-year 2024 --test-end-year 2025
```

Shared cutoff flags (all three scripts): `--train-end-year`, `--val-year` / `--no-val`, `--test-start-year`, `--test-end-year`, `--out-base`, `--group-size`.

### Geographic splits (unchanged; not forecasting)

| Dataset | Script | Dir |
|---------|--------|-----|
| H1N1 geo | `prepare_h1n1_geo.py` | `data/h1n1` |
| COVID geo | `prepare_covid_geo.py` | `data/covid` |

---

## B) COVID clade holdout (leaf recovery)

Goal: hold out one clade of leaves per tree; train without those leaves; eval recovery of the held-out clade (same metrics as H1N1 leafholdout).

| Step | Command |
|------|---------|
| Spike extract | `sbatch scripts/slurm_covid_extract.sh` |
| Clade labels (preferred) | `python scripts/prepare_covid_cladeholdout.py --write-clade-tsv` |
| Or pass TSV | `... --clade-tsv data/covid/train/africa_clades.tsv ...` |
| Year-proxy fallback | `... --mode majority-year` (weak proxy when Nextclade unavailable) |
| Tree build | `sbatch scripts/slurm_covid_cladeholdout_pipeline.sh` |
| Eval | `python scripts/eval_leaf_holdout.py --data data/covid_cladeholdout --checkpoint … --max-seq-len 1280` |

**Output dir:** `data/covid_cladeholdout/`

```
data/covid_cladeholdout/
  train|val|test/covidch{split}_group_NNN.fasta   # reduced trees (clade removed)
  heldout/covidch_{split}_group_NNN_heldout.fasta # held-out clade leaves
  heldout/covidch_{split}_group_NNN_heldout_meta.json
  SPLIT_PROTOCOL.json
```

**Clade selection (default):** among Nextclade `clade` labels present in the chunk, pick the clade whose size is closest to `--holdout-frac-target` (default 0.15), that is not ≥85% of the tree, and that leaves ≥ `--min-group` train leaves. Override with `--holdout-clade NAME`.

**Eval reuse:** `eval_leaf_holdout.py` resolves any `*_{split}_group_{NNN}_heldout.fasta` under `heldout/`, so H1N1 (`h1n1lh_…`) and COVID (`covidch_…`) share one eval script.

---

## Expected data dirs (summary)

| Dir | Role |
|-----|------|
| `data/h3n2/{train,val,test}` | H3N2 temporal (default cutoffs) |
| `data/h3n2_temporal_forecast/` | H3N2 alternate forecast cutoffs |
| `data/h1n1_temporal/` | H1N1 temporal forecast |
| `data/covid_temporal/` | COVID temporal forecast |
| `data/covid_cladeholdout/` | COVID clade leaf-recovery |
| `data/h1n1_leafholdout/` | H1N1 random leaf-holdout (existing) |
| `data/h1n1/`, `data/covid/` | Geographic splits (existing) |

After prep → `run_all_groups` / SLURM pipeline → `precompute_plm.py` / `precompute_ref_rates.py` → `train.py` → domain eval.

---

## D) Pan-flu forecast generality

**Question:** Does cross-strain training improve forecasting on a held-out subtype/season?

| Script | Role |
|--------|------|
| `scripts/prepare_panflu_pool.py` | Merge H3N2 + H1N1 (+ FluB) inventory with `subtype=` header tags |
| `scripts/prepare_panflu_forecast.py` | Train-pool ablation × calendar vs NH-season test |
| `scripts/betty_prep_panflu_forecast.sh` | Build matrix dirs B0–G6 on Betty |
| `scripts/betty_submit_panflu_forecast.sh` | Fresh ckpts per row |

**Train-pool ladder:** `h3n2` → `h3n2_h1n1` → `panflu`

**Test holdouts:**

| Mode | Rule | Example |
|------|------|---------|
| Calendar | train year ≤2019; test = test-subtype in 2020 | H3N2 tips in 2020 |
| NH season | train seasons < `2019-2020`; test = that season | Full Oct19–Sep20 tree |

**Matrix rows (checkpoint dir names in `SPLIT_PROTOCOL.json`):**

| Row | Train pool | Test |
|-----|------------|------|
| B0 | H3N2 only | H3N2 cal 2020 → `h3n2_only_forecast_2020_cal` |
| B1 | H3N2 only | H3N2 season 2019–20 → `h3n2_only_forecast_2020_season` |
| G1 | H3N2+H1N1 | H3N2 cal 2020 |
| G2 | H3N2+H1N1 | H1N1 cal 2020 |
| G3 | Pan (H3N2+H1N1+FluB) | H3N2 cal 2020 |
| G4 | Pan | H3N2 season 2019–20 |
| G5 | Pan | H1N1 cal 2020 |
| G6 | Pan | FluB cal 2020 |

Eval: `benchmarks/coverage_curves.py` + [`eval_everest_forecasting.py`](../scripts/eval_everest_forecasting.py) — see [`EVEREST_EVAL.md`](EVEREST_EVAL.md).

**Caption discipline:** Calendar vs NH-season are separate rows; always report train-pool size (# seqs, # subtypes, # trees).

---

## C) Epidemic-aware trees (v2 retrain wave)


Full plan: [`EPIDEMIC_TREE_SPLITS.md`](EPIDEMIC_TREE_SPLITS.md).
