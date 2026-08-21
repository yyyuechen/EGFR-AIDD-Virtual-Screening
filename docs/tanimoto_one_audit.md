# Tanimoto = 1.0 Audit

## Purpose

The Morgan radius study found validation molecules with
`max_train_tanimoto = 1.0` even though the standard Murcko scaffold split
has zero scaffold overlap. Before any further tuning, this audit checks
whether those cases are true molecule-level leakage or a fingerprint
representation effect.

## Reference fingerprint generator settings

The audit uses the exact project reference fingerprint:

| Setting | Value |
|---|---|
| API | `rdkit.Chem.AllChem.GetMorganGenerator` |
| Radius | 2 |
| fpSize | 2048 |
| `includeChirality` | False |
| `useBondTypes` | True |
| `countSimulation` | False |
| `onlyNonzeroInvariants` | False |
| `includeRingMembership` | True |
| Fingerprint type | `ExplicitBitVect` (binary bit vector) |

The generator produces binary bit vectors; count simulation is off and
chirality is not encoded.

## Results

* Affected validation molecules: **15 / 1,015**
* Pairs with Tanimoto = 1.0: **27**
* Exact canonical SMILES identical: **0 / 27**
* InChIKey identical: **0 / 27**
* Murcko scaffold identical: **0 / 27**
* Classification: **all 27 pairs are different structures**

Representative pairs:

| Validation ID | Training ID | Difference |
|---|---|---|
| CHEMBL2110732 | CHEMBL3680378 | alkyl chain length / ring size outside radius-2 environment |
| CHEMBL3360610 | CHEMBL3360609 | side-chain length `OCCCCCC` vs `OCCCCC` |
| CHEMBL3360610 | CHEMBL3355882 | ring attachment position `cc` vs `ccc` |

## Answers to the audit questions

1. **Is there exact molecule-level train/validation leakage?**
   No. Every pair has different canonical SMILES, different InChIKey, and
   different Murcko scaffold. No exact molecule crosses the split.

2. **Are the Tanimoto=1 cases different structures collapsing to the same
   binary fingerprint?**
   Yes. All 27 pairs are different chemical structures whose radius-2,
   2048-bit binary Morgan fingerprints are identical.

3. **Are stereochemical differences being ignored?**
   The generator is configured with `includeChirality = False`, so chirality
   is not encoded in the reference fingerprint. However, none of the 27
   observed pairs are stereoisomer pairs; stereochemistry is a general
   limitation of the representation, but it is not the direct cause of these
   particular Tanimoto=1 cases.

4. **Is this simply a consequence of fingerprint hashing / representation
   limitations?**
   Yes. The differing atoms in these pairs lie outside the radius-2 circular
   environments (or collapse through the 2048-bit hashed binary mapping), so
   distinct molecules map to the same fingerprint.

## Conclusion

No molecule-level leakage is present. The Tanimoto=1 cases are caused by the
Morgan fingerprint representation collapsing different structures to the
same binary bit pattern. No split or data reassignment is needed, and the
controlled nBits study can proceed.

## Files

```text
src/models/tanimoto_one_audit.py
results/tanimoto_one_audit.csv
results/tanimoto_one_audit.json
```
