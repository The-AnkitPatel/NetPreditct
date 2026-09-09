---
tags: [ml, leakage, cross-validation]
created: 2026-09-09
---
# Temporal Leakage Prevention

Network data is autocorrelated. Random K-Fold cross-validation will cause catastrophic data leakage, making the model look perfect in the lab but fail in production.

## Strict Walk-Forward Cross-Validation
We train on $[0, T]$ and test on $[T + gap, T + gap + window]$.

> [!warning] The Embargo Gap
> We must insert an *embargo gap* between the training and validation sets to ensure the model cannot "cheat" by memorizing recent state.

```ascii
Time --------------------------------------------->
[--- Train ---] [ Gap ] [--- Val ---]
```

We also apply causal feature engineering (only using information strictly available at time $t$).
