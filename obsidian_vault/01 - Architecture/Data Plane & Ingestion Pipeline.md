---
tags: [architecture, data-plane, ingestion]
created: 2026-09-09
---
# Data Plane & Ingestion Pipeline

The ingestion pipeline processes INT (In-band Network Telemetry) and SNMP traps at high velocity.

> [!warning] Bottleneck Alert
> Standard polling is too slow. We use asynchronous workers and a ring buffer to handle bursts up to 10k eps (events per second).

## Ring Buffer Approach
We utilize a sliding window ring buffer built on `collections.deque` with a fixed `maxlen`. This guarantees $O(1)$ appends and pops.

```python
from collections import deque

class TelemetryBuffer:
    def __init__(self, window_size: int = 1000):
        self.buffer = deque(maxlen=window_size)

    def ingest(self, metric: dict):
        # O(1) append; oldest element is dropped if full
        self.buffer.append(metric)
```

See [[Sliding Window State Machine]] for the temporal slicing logic.
