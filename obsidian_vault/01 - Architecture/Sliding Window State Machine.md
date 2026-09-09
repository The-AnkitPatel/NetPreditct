---
tags: [architecture, state-machine, optimization]
created: 2026-09-09
---
# Sliding Window State Machine

Predicting network failures requires looking at a window of history $W = [t-k, t]$. 

## Temporal Window Slicing
Instead of recomputing aggregations over the entire window for every new packet, we maintain rolling states (sum, max, min).

> [!info] O(1) Rolling Updates
> $S_{t} = S_{t-1} + x_t - x_{t-W}$

```mermaid
sequenceDiagram
    participant Stream as Stream
    participant Buffer as O(1) Buffer
    participant ML as Model
    Stream->>Buffer: Push metric (t)
    Buffer->>Buffer: Evict metric (t-W)
    Buffer->>Buffer: Update rolling mean/var
    Buffer->>ML: Send state vector
```

This feeds directly into our [[Multi-Horizon LightGBM]] model for T+n predictions.
