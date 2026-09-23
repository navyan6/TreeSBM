# Epidemic-aware tree splits (cross-virus upgrade)

TreeSBM learns branch substitution patterns on **FastTree topologies**. When a group mixes unrelated transmission chains, the tree is biologically wrong and metrics are hard to interpret. This doc plans a **v2 retrain wave** with coherent trees across all paper viruses.


---

## Should we retrain flu / COVID / HIV?

**Yes — for a unified paper story**, but **phased**:

| Virus | Current tree unit | Problem | v2 unit | Retrain priority |
|-------|-------------------|---------|---------|------------------|
| **Filovirus L** | date chunks | mixed outbreaks | **outbreak** | **Paused** (metadata in git; no SLURM train) |
| **COVID Spike** | country × date chunk | OK-ish locally | **country × Pango lineage** or WHO clade | **P1** |
| **H3N2 / H1N1 HA** | year-band date chunks | not true seasons | **NH season** (Oct(Y−1)–Sep(Y)) or country×season | **P1** |
| **HIV env** | geo country pools | weak transmission | **country × subtype cluster** (or LANL cluster) | **P2** |

Retraining everything is **worth it** if the paper claims TreeSBM models **epidemic evolution on real transmission trees**. Keeping old ckpts as ablation rows (“chunk trees”) strengthens the narrative.

Expected effect:
- **mut_recovery / Coverage@K:** modest ↑ when topology matches transmission; not guaranteed on all metrics.
- **Interpretability:** large ↑ — reviewers can understand what each tree is.

Compute: ~1 train job per domain × (lit + no-lit). Active wave: **COVID + flu epidemic only** (`bash scripts/betty_submit_epidemic_retrain.sh`).

---

## COVID Spike v2 (scripts ready)

**Script:** `scripts/prepare_covid_epidemic.py`  
**Prep on Betty:** `bash scripts/betty_prep_epidemic_splits.sh` (after `slurm_covid_extract.sh`)

- Group key: `(country, Nextclade clade)`; fallback `(country, year)`
- Temporal cutoffs unchanged (train ≤2022, test 2024–2025)
- **Output:** `data/covid_epidemic/`; ckpt `covid_v6_epidemic_mutrec`

---

## Influenza HA v2 (scripts ready)

**Scripts:** `scripts/prepare_h3n2_epidemic.py`, `scripts/prepare_h1n1_epidemic.py`

- H3N2: one tree per **NH flu season** (`2019-2020`, …)
- H1N1: one tree per **(location|country, season)**
- **Output:** `data/h3n2_epidemic/`, `data/h1n1_epidemic/`
- **Ckpts:** `h3n2_v4_epidemic_mutrec`, `h1n1_v3_epidemic_mutrec`

Audit: `python scripts/audit_epidemic_splits.py`

---

## HIV env v2 (planned, lower priority)

**Script (to add):** `scripts/prepare_hiv_cluster.py`

- Use LANL cluster or country×year window (≤2 yr) like filo fallback.
- Holdout geography unchanged for Table comparability.
- **Output:** `data/hiv_geo_epidemic/`; ckpt `hiv_geo_v2_epidemic_mutrec`.

---

## Migration checklist

1. Implement prep script + `SPLIT_PROTOCOL.json` with outbreak/season/clade histogram.
2. Add `audit_*_splits.py` gate (`ready_to_train`, non-empty val).
3. SLURM pipeline on new `DATA_ROOT` + new `CKPT_DIR`.
4. Re-run Coverage@K + mut_recovery; paste to new table row **v2 epidemic trees**.
5. Keep v1 numbers in appendix as “date-chunk ablation”.

---

## Paper wording (one sentence)

> Training trees group sequences by **transmission unit** (influenza season, COVID locality×lineage, filovirus outbreak); temporal holdouts evaluate forecasting on later seasons or unseen outbreaks.
