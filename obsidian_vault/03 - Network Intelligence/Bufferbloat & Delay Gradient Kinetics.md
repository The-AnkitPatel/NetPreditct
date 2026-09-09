---
tags: [network, bufferbloat, physics]
created: 2026-09-09
---
# Bufferbloat & Delay Gradient Kinetics

Bufferbloat occurs when unnecessarily large queues cause high latency and jitter, defeating TCP's congestion avoidance.

## Queuing Delay Physics
The delay gradient $\frac{\partial d}{\partial t}$ is a leading indicator of congestion. 

> [!warning] Drop Cliff Triggers
> When the buffer fills, tail-drop occurs, leading to synchronized TCP window collapse. 

$$ \Delta d = RTT_{current} - RTT_{min} $$

By tracking the derivative of RTT over our [[Sliding Window State Machine]], we can detect the onset of bufferbloat before packets are dropped.
