# Final Candidate Prioritization

## 1. Eligible candidate pool

* Source: the 64-molecule M6.5 pool (`results/m65_improved_shortlist.csv`)
* Original N = 64
* Known positive controls excluded = 4 (afatinib, erlotinib, gefitinib,
  lapatinib)
* Exact training-set matches excluded = 0 among the remaining 60
* **Final eligible N = 60**

The canonical final-candidate ranking is in
`results/final_known_inhibitor_ranking.csv`; model evidence uses the locked
Morgan-RF (`radius=2, nBits=2048, max_features=0.20`).

## 2. Evidence tiers

Tiers are assigned with explicit rules, not a weighted score:

* potential: HIGH if canonical prediction >= 6.0 or eligible rank <= 10;
  MODERATE if prediction >= 5.6; else LOW;
* stability: GOOD if median rank <= 15 and top-10 frequency >= 0.5;
  MODERATE if median rank <= 30 and top-10 frequency >= 0.25; else WEAK;
* applicability: HIGH (sim >= 0.8), MODERATE (0.6-0.8), LOWER (0.4-0.6),
  OOD (<0.4);
* Tier 1: HIGH potential + GOOD stability + acceptable AD + physicochemical
  pass + supportive/moderate docking;
* Tier 2: novel-but-supported (high potential + lower/OOD AD + physchem
  pass + non-concerning docking), docking-pending (acceptable AD + physchem
  pass, docking not yet performed), or moderate-potential-but-well-supported.

## 3. Final shortlist

### Tier 1 (highest priority for experimental testing)

| candidate | pred pIC50 | rank | median rank (25 runs) | top10 freq | max sim | AD | physchem | docking |
|---|---:|---:|---:|---:|---:|---|---|---:|
| CHEMBL13983 | 5.93 | 4 | 9.0 (3-27) | 0.56 | 0.64 | MODERATE | pass | SUPPORTIVE (-9.22) |

Tier 1 is intentionally small: CHEMBL13983 is the only eligible candidate
that converges across prediction, ranking stability, applicability domain,
physicochemical properties, and docking.

### Tier 2 (exploratory / novel / evidence-incomplete)

| candidate | pred pIC50 | rank | median rank (25 runs) | top10 freq | max sim | AD | physchem | docking |
|---|---:|---:|---:|---:|---:|---|---|---:|
| CHEMBL5564149 | 6.12 | 3 | 2.0 (2-5) | 1.00 | 0.31 | OOD | pass | SUPPORTIVE (-9.08) |
| CHEMBL131921 | 5.87 | 7 | 12.0 (3-27) | 0.48 | 0.75 | MODERATE | pass | INCOMPLETE |
| CHEMBL2315700 | 5.84 | 8 | 9.0 (4-35) | 0.64 | 0.17 | OOD | pass | INCOMPLETE |
| CHEMBL243664 | 5.71 | 14 | 10.0 (2-35) | 0.52 | 0.58 | LOWER | pass | SUPPORTIVE (-8.64) |

Priority order within Tier 2: P2 CHEMBL5564149, P3 CHEMBL131921,
P4 CHEMBL2315700, P5 CHEMBL243664.

## 4. Why each compound was selected

* CHEMBL13983: the only candidate with simultaneously moderate
  applicability-domain support, physicochemical pass, strong docking, and
  stable ranking (top-10 in 56% of 25 frozen runs). Highest overall
  prioritization confidence.
* CHEMBL5564149: highest predicted pIC50 (6.12) among shortlisted
  candidates and extremely stable (median rank 2, top-10 100%), with a
  supportive docking score; its low max similarity makes it an
  exploratory/high-risk-high-reward option.
* CHEMBL131921: high candidate rank (7), good applicability-domain support
  (sim 0.75), physchem pass, but docking has not been performed, so it is
  Tier 2 with docking pending.
* CHEMBL2315700: stable top-10 (64%) and physchem pass with very low
  training similarity (0.17), representing a novel/uncertain option that
  has not yet been docked.
* CHEMBL243664: moderate potential, stable ranking, physchem pass, and a
  supportive docking score (-8.64), with only lower applicability-domain
  support.

## 5. Notable exclusions

The following high-predicted candidates were NOT shortlisted because they
fail the existing physicochemical filter (Lipinski/Veber):

| candidate | pred pIC50 | rank | median rank | max sim | ADMET_ok |
|---|---:|---:|---:|---:|---:|
| CHEMBL2316916 | 6.64 | 1 | 1.0 | 0.30 | False |
| CHEMBL5614250 | 6.16 | 2 | 4.0 | 0.43 | False |
| CHEMBL554993 | 5.91 | 5 | 5.0 | 0.31 | False |
| CHEMBL512054 | 5.90 | 6 | 4.0 | 0.33 | False |

CHEMBL2316916 is the most striking exclusion: highest predicted activity and
perfect rank stability, but it fails the project's physicochemical filter,
has strong OOD status, and has no external evidence. It is reported as
deprioritized, not hidden.

## 6. Major uncertainties

* Only 60 eligible non-training candidates, and only 16 have docking
  evidence.
* No repository-level external evidence exists for these candidate-like
  molecules; they are computational hypotheses only.
* Docking evidence is score-based (Vina), not pose-validated for all
  candidates.
* OOD candidates have historical MAE ~1.07 and therefore larger expected
  quantitative error.
* Physicochemical filtering is rule-based, not experimental ADMET.
* No uncalibrated success probabilities are claimed.

## 7. If only ONE assay slot existed

Test **CHEMBL13983** first: it has the best balance of predicted activity,
stable ranking, moderate applicability domain, physicochemical pass, and
supportive docking.

## 8. If FIVE assay slots existed

Test the full shortlist in priority order:

1. CHEMBL13983
2. CHEMBL5564149
3. CHEMBL131921
4. CHEMBL2315700
5. CHEMBL243664

## 9. Best balance of predicted activity, confidence, novelty

**CHEMBL13983** is the best balanced choice. **CHEMBL5564149** offers higher
predicted activity but at the cost of OOD applicability domain.

## 10. Most interesting high-risk / high-reward option

**CHEMBL5564149**: highest predicted activity in the shortlist (6.12),
extremely stable ranking, physchem pass, supportive docking, but very low
training similarity (0.31). It is the strongest candidate to test whether
the model transfers to genuinely unfamiliar chemistry.

## 11. Required experimental evidence

For any shortlisted candidate, the next experimental evidence should be:

1. biochemical EGFR inhibition assay;
2. quantitative IC50 measurement with replicates;
3. selectivity / off-target follow-up if activity is confirmed;
4. cellular efficacy follow-up only after biochemical confirmation.

No detailed wet-lab protocol is designed here.

## Files

```text
src/models/final_candidate_prioritization.py
results/final_candidate_stability_25runs.csv
results/final_candidate_evidence_matrix.csv
results/final_candidate_shortlist.csv
results/final_candidate_prioritization_summary.json
results/figures/final_candidate_potential_vs_confidence.png
results/figures/final_candidate_rank_stability.png
```

Historical M6/M7/M8 ranking files were not modified.
