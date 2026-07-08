# TODO: Expand to Full 7-Tier Mechanistic Validity Encoding

## Current state

The psych ORC paper uses a 5-tier encoding in `experiments/catalog_data.py`:

```python
VERDICTS = {
    "Validated": 5,
    "Mechanistically Supported": 4,
    "Causally Suggestive": 3,
    "Underdetermined": 2,
    "Inconclusive": 2,
    "Disconfirmed": 1,
}
```

This is missing two tiers from the Mechanistic Validity framework (`mechanistic-validity/src/mechval/models.py`, `VerdictTier` enum):
- **Proposed** (no intervention evidence)
- **Triangulated** (all internal + external + construct validity criteria met)

## Target encoding

Full 7-tier scale matching the canonical Mechanistic Validity framework. Higher ordinal = better verdict:

| Ordinal | Tier |
|---------|------|
| 1 | Disconfirmed |
| 2 | Underdetermined |
| 3 | Proposed |
| 4 | Causally Suggestive |
| 5 | Mechanistically Supported |
| 6 | Triangulated |
| 7 | Validated |

This matches the chemistry-validity-audit repo's encoding (implemented 2026-07-07, deviation D5).

## What needs to change

1. **`experiments/catalog_data.py`** — Update `VERDICTS` dict to 7-tier encoding
2. **Re-audit the 44 entries** — Check if any should be Proposed or Triangulated
   - "Inconclusive" currently maps to Underdetermined (ordinal 2). Decide if any Inconclusive entries are actually Proposed (ordinal 3) per MechVal criteria
   - Check if any Validated entries meet Triangulated but not full Validated criteria
3. **Survival threshold** — Change from ordinal >= 3 to ordinal >= 4 (Causally Suggestive and above)
4. **ORC analysis scripts** — Re-run with updated encoding
5. **Paper text** — Update any references to the 5-tier scale

## Impact on results

- If no entries change tiers (just ordinal remapping), ORC curvatures and z-scores are unchanged since ORC operates on graph topology, not ordinal values
- If entries get re-classified (Inconclusive -> Proposed, or Validated -> Triangulated), the tier-stratified analyses (H7 equivalent) would change
- The survival binary should be preserved for most entries since the threshold moves proportionally

## Priority

Low-medium. The paper is submitted to GigaScience as of 2026-07-08. This would go into a revision or v2. The chemistry paper already uses 7 tiers, so cross-study comparability improves with this change.
