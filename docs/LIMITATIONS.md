# Limitations

## Retrospective Only

All results are retrospective. There is no prospective or wet-lab
validation, and no compound in this repository is reported as a confirmed
EGFR inhibitor. Final candidates are hypotheses for experimental testing.

## External Transferability

The external set contains only 59 heterogeneous ChEMBL IC50 records and is
a stress test, not a gold standard. External performance is substantially
weaker than internal scaffold-aware performance: the RandomForest external
Spearman mean is +0.206 (5/5 seeds positive), while single-seed ensemble
gains were not reproducible across seeds.

## Split and Applicability-Domain Caveats

* The standard scaffold split is one deterministic realization; other
  partitions shift the numbers, although the tuned model improved over the
  baseline in 5/5 alternative whole-Murcko-scaffold partitions.
* The hard scaffold OOD split is deliberately extreme: every test scaffold
  is a singleton, so it represents worst-case extrapolation rather than a
  typical scaffold-aware benchmark.
* Morgan Tanimoto similarity is representation-dependent; structurally
  similar molecules can still show activity cliffs, so applicability-domain
  support is a guide, not a guarantee.

## Data and Labels

* Assay heterogeneity affects activity labels; the pipeline mitigates this
  by keeping exact IC50 measurements only and aggregating with the median.
* The candidate library is derived from the same ChEMBL source as the
  training data and may carry related chemistry or assay bias.

## Docking and ADMET

* AutoDock Vina scores are screening approximations, not true binding
  affinities; the receptor is treated as rigid.
* Docking pose validation uses one co-crystal (erlotinib in 1M17) as a
  setup sanity check.
* ADMET descriptors are rule-based estimates (Lipinski, Veber, ESOL, QED),
  not experimental measurements.

## Model-Selection Context

Hyperparameter decisions were made sequentially on one fixed scaffold
validation partition, so repeated adaptive use of that validation split may
introduce some model-selection bias. To limit this, the final locked model
was evaluated once on the untouched standard-scaffold test set, and the
feature-subsampling benefit was confirmed across five RF seeds.
