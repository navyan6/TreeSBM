# Data splits

TreeSBM forecasting and holdout data are prepared by the scripts below.
Each prep writes a `SPLIT_PROTOCOL.json` under the output directory.

## Temporal splits

Train on sequences before a cutoff year; evaluate on later years.

| Dataset | Prep script | Default cutoffs | Output |
|---------|-------------|-----------------|--------|
| H3N2 HA | `scripts/prepare_h3n2_temporal.py` | train≤2022 / val=2023 / test=2024 | `data/h3n2` |
| H1N1 HA | `scripts/prepare_h1n1_temporal.py` | train≤2023 / val=2024 / test=2025 | `data/h1n1_temporal` |
| COVID Spike | `scripts/prepare_covid_temporal.py` | train≤2022 / val=2023 / test=2024–2025 | `data/covid_temporal` |

**Grouping.** Flu groups are date-ordered chunks of `--group-size` sequences within one year band (train / val / test); a group never crosses the cutoff. Cutoffs are collection-year based. COVID groups are single-country, date-ordered chunks within the same year bands.

```bash
python scripts/prepare_h3n2_temporal.py
python scripts/prepare_h1n1_temporal.py
# COVID: extract Spike FASTAs first, then
python scripts/prepare_covid_temporal.py
```

Shared flags: `--train-end-year`, `--val-year` / `--no-val`, `--test-start-year`, `--test-end-year`, `--out-base`, `--group-size`.

## Geographic splits

| Dataset | Script | Dir |
|---------|--------|-----|
| H1N1 | `scripts/prepare_h1n1_geo.py` | `data/h1n1` |
| COVID | `scripts/prepare_covid_geo.py` | `data/covid` |
| HIV Env | `scripts/prepare_hiv_splits.py` | `data/hiv_geo`, `data/hiv_temporal` |

## Leaf / clade holdout

| Dataset | Script | Dir |
|---------|--------|-----|
| H1N1 random leaf holdout | `scripts/prepare_h1n1_leafholdout.py` | `data/h1n1_leafholdout` |
| COVID clade holdout | `scripts/prepare_covid_cladeholdout.py` | `data/covid_cladeholdout` |

Held-out leaves are written under `heldout/`. Evaluate with:

```bash
python scripts/eval_leaf_holdout.py --data data/h1n1_leafholdout --checkpoint … --max-seq-len 566
```

## Epidemic splits

Outbreak / season-scoped trees for COVID, H3N2, H1N1, and HIV:

| Dataset | Script | Dir |
|---------|--------|-----|
| COVID | `scripts/prepare_covid_epidemic.py` | `data/covid_epidemic` |
| H3N2 | `scripts/prepare_h3n2_epidemic.py` | `data/h3n2_epidemic` |
| H1N1 | `scripts/prepare_h1n1_epidemic.py` | `data/h1n1_epidemic` |
| HIV | `scripts/prepare_hiv_epidemic.py` | `data/hiv_epidemic` |

Shared helpers: `scripts/epidemic_split_common.py`, `scripts/audit_epidemic_splits.py`.

## Pipeline after prep

```bash
python scripts/run_all_groups.py --data <split_dir>
python scripts/precompute_plm.py --data <split_dir>
python scripts/precompute_ref_rates.py --data <split_dir>
python scripts/train.py --data <split_dir> …
```
