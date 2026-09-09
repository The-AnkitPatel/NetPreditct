---
aliases: [Engineering Diary, Dev Diary, NetPredict Genesis]
tags: [diary, reflections, engineering, telemetry, machine-learning, networking]
created: 2026-09-09
updated: 2026-09-09
author: Ankit Patel (3rd Year B.Tech CSE/IT)
---

# 📓 Dev Diary: Building NetPredict

*A chronicled descent into 3:45 AM hostel room delirium, packet queues, mathematical illuminations, and the obsessive quest to forecast network failure before the first packet drops.*

---

## 🕒 03:42 AM — The Scent of Solder, Cold Coffee, and Dropped Packets

Every junior in my college is building another CRUD wrapper around an LLM API. I couldn’t do it. If I was going to spend my 3rd-year B.Tech burning sleep, I wanted to build something that touches bare metal and mathematical reality—something hyperscalers like Cloudflare, Meta, or Google SREs actually sweat over.

The project started with a personal vendetta against bufferbloat on our campus LAN. Whenever someone started pulling a massive dataset or torrenting in the wing, everyone else’s SSH sessions hung, VoIP jittered into robot noises, and gaming ping spiked from 14ms to 1,800ms. Traditional SNMP monitoring was useless: it polled every 30 or 60 seconds. By the time an alert popped saying *"Interface utilization at 98%"*, the queue had already experienced tail-drop collapse, TCP Reno/Cubic had halved their congestion windows, and the network had fallen off a cliff.

Why detect a disaster after it happens? Why not predict the crash **5, 15, or 30 minutes in advance**? That question became **NetPredict**.

```
[Raw INT / SNMP Telemetry] ──> [O(1) Ring Buffer] ──> [dRTT/dt Kinetics] 
                                                               │
   ┌───────────────────────────────────────────────────────────┘
   ▼
[Multi-Horizon LightGBM] ──> [Isotonic Calibrator] ──> [Conformal Intervals] ──> [TreeSHAP Root Cause]
```

---

## ⚡ The Death of the Recurrent Delusion: Why GBDTs Crushed LSTMs

In October, I fell head-over-heels into deep learning dogma. *"It's sequential telemetry, so of course you need an LSTM or a Temporal Fusion Transformer!"*

I burned three straight weeks of compute on my poor GTX 1660 Ti laptop training stacked LSTM networks. The results were atrocious:
1. **Categorical tabular friction**: Real telemetry mixes continuous gauges (`rtt`, `queue_depth`, `bytes_in`) with high-cardinality discrete identifiers (`interface_index`, `vlan_id`, `egress_port_id`). LSTMs required complex entity embeddings and struggled with missing SNMP metrics.
2. **Computational Bloat**: Inference latency on an LSTM hovered at ~18ms per batch. When you're ingesting 10,000 events/sec via our [[Data Plane & Ingestion Pipeline]], an 18ms forward pass creates an catastrophic memory backlog.
3. **Vanishing Gradient over Long Horizons**: Predicting $T+30\text{m}$ required unfolding recurrent cells across thousands of micro-steps.

The epiphany came when I re-framed the problem: **the network state is Markovian over a sufficiently rich rolling window**. By engineering an $O(1)$ ring buffer in our [[Sliding Window State Machine]], we could extract explicit kinetic primitives—exponential moving averages, sliding variances, and rate derivatives—and feed them to **Gradient Boosted Decision Trees**.

We deployed [[Multi-Horizon LightGBM]]. Inference plummeted from 18ms to **0.14ms**. It ingested raw categorical features natively, ignored null values gracefully, and allowed independent loss optimization for each distinct forecasting horizon ($T+5\text{m}, T+15\text{m}, T+30\text{m}$).

---

## 🌊 The Physics of Bufferbloat: Hunting the Delay Gradient $\frac{dRTT}{dt}$

Most network monitoring tools watch queue depth $Q(t)$ as a scalar. But in high-speed switches, buffers don’t fill linearly—they undergo phase transitions governed by queuing kinetics.

```
       Delay (RTT)
           ▲
           │                                 Tail-Drop Cliff (Packets Lost)
           │                                       /
           │                                     ┌─┘
           │                                   ┌─┘
           │                         Phase 2: ┌┘  dRTT/dt >> 0 (Queue filling)
           │               ──────────┐       /
           │   Phase 1:   /          └──────┘
           │   Nominal   /
           └────────────┴──────────────────────────────────────► Time
                      Throughput Saturation (Kleinrock Knee)
```

The queuing delay component is isolated as:
$$\Delta d_t = RTT_t - RTT_{min}$$

Where $RTT_{min}$ represents the physical propagation delay (speed of light in glass + fixed serialization). When traffic reaches link capacity $C$, any additional ingress rate $\lambda > C$ accumulates in the buffer:
$$\frac{dQ}{dt} = \lambda(t) - C \implies \frac{dRTT}{dt} = \frac{1}{C}\left(\lambda(t) - C\right)$$

> [!important] The Mathematical Breakthrough
> By tracking the **delay gradient** $\frac{\partial (RTT)}{\partial t}$ across multi-scale sliding windows (10s, 30s, 60s), our LightGBM model detects buffer saturation **hundreds of seconds before tail-drop packet loss occurs**. The gradient is the leading indicator; packet drop is merely the funeral notice. See [[Bufferbloat & Delay Gradient Kinetics]].

---

## 💔 The 99.4% Heartbreak: Eliminating Temporal Leakage

On a Tuesday at 2:15 AM, I stared at my screen in disbelief. My first LightGBM model scored an ROC-AUC of **0.994** on $T+15\text{m}$ link failure prediction. I was ready to draft a submission to ACM SIGCOMM. I thought I was a prodigy.

Then, cold reality hit.

I looked at the cross-validation script. I had used `sklearn.model_selection.KFold(shuffle=True)`. In time-series telemetry, shuffling is suicide. Network state is violently autocorrelated: packet queue depth at $t=100$ is 99% correlated with $t=101$. By randomly shuffling samples, my validation fold was interpolating between $t-1$ and $t+1$. The model wasn’t predicting the future—it was memorizing the past.

When I refactored to a strict **Walk-Forward Validation** scheme with an **Embargo Gap** ($\tau_{embargo} \ge \text{Horizon} + 3\times \tau_{decay}$), my glorious 99.4% ROC-AUC collapsed to a cold, brutal **0.784**.

```
Walk-Forward Validation Architecture:
Time ───────────────────────────────────────────────────────────────►
Fold 1: [=== Train T0 ===] [ Embargo Gap ] [=== Val T1 ===]
Fold 2: [====== Train T0..T1 ======] [ Embargo Gap ] [=== Val T2 ===]
```

I wanted to throw my laptop out the hostel window. But that 0.784 was *honest*. It proved our features were learning actual predictive signal rather than temporal artifacts. We documented this rigorously in [[Temporal Leakage Prevention]].

---

## 🎯 Calibration Sickness: Why Raw Tree Probabilities Lie

Once the model was clean, another flaw emerged. In testing, an alert with `p_predicted = 0.85` only failed 40% of the time. Conversely, scores around `0.30` were occasionally triggering catastrophic brownouts.

Decision trees optimize split criteria (e.g., Gini or Logloss) by pushing predictions toward leaf extremes ($0$ or $1$) or clustering around step thresholds. **Raw tree leaf frequencies are ranking scores, not true physical probabilities.** If an operator is asked to reroute a 100 Gbps core trunk, "Score 0.85" is a dangerous gamble.

We implemented **Isotonic Regression** on an out-of-time calibration set:
$$\min_{m} \sum_{i=1}^{n} \left( y_i - m(\hat{p}_i) \right)^2 \quad \text{subject to } m(\hat{p}_a) \le m(\hat{p}_b) \ \forall \ \hat{p}_a \le \hat{p}_b$$

```
   1.0 ┌                                         / (Ideal Calibration)
       │                                     x  /
   P   │                                 x     / 
   (y) │                             x        /   * Raw LightGBM (Sigmoidal warp)
       │                         x           /    x Calibrated via Isotonic
       │                   x                /
   0.0 └───────────────x───────────────────/──────────────────────────►
      0.0                                1.0     Predicted Probability p
```

The transformation was night and day. The Brier score dropped from $0.284 \to 0.061$:
$$Brier = \frac{1}{N} \sum_{i=1}^N (f_i - y_i)^2$$
Now, when NetPredict outputs an 80% failure probability, exactly 8 out of 10 historical links in that bucket experienced failure. See [[Probability Calibration & Brier Score]].

---

## 🛡️ Beyond Point Estimates: Split Conformal Prediction

Telling an on-call engineer *"RTT will be 142.3 ms at T+15m"* is reckless. During a traffic surge, network dynamics are stochastic. Point forecasts give a false sense of security.

I wanted mathematically rigorous prediction intervals without making fragile Gaussian assumptions. Enter **Split Conformal Prediction**:

1. Partition historical holdout data into proper training and calibration sets $(X_{cal}, Y_{cal})$ of size $n$.
2. Compute non-conformity residuals on the calibration set:
   $$R_i = |Y_i - \hat{\mu}(X_i)|$$
3. Compute the empirical quantile $\hat{q}$ at significance level $\alpha = 0.10$:
   $$\hat{q} = \text{Quantile}\left( R_{1:n}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$
4. For any live incoming telemetry vector $X_{test}$, construct the interval:
   $$C(X_{test}) = \left[ \hat{\mu}(X_{test}) - \hat{q}, \quad \hat{\mu}(X_{test}) + \hat{q} \right]$$

$$P(Y_{test} \in C(X_{test})) \ge 1 - \alpha = 90\%$$

This yields **distribution-free, finite-sample marginal coverage guarantees**. Whether the network faces an MTU black hole or a DDoS flash crowd, the true latency falls within our interval at least 90% of the time. See [[Split Conformal Prediction]].

---

## 🏛️ The Pride of an Engineering Marvel

Looking at NetPredict running on my monitor right now is surreal:
- **FastAPI backend** asynchronously ingesting synthetic and real PCAP streams through an in-memory lock-free ring buffer ([[Data Plane & Ingestion Pipeline]]).
- **Sliding Window Engine** extracting 48 temporal features in $O(1)$ time per packet tick ([[Sliding Window State Machine]]).
- **Multi-Horizon Engine** generating calibrated failure probabilities across 5, 15, and 30-minute horizons ([[Multi-Horizon LightGBM]]).
- **TreeSHAP** decomposing every alert into exact additive contributions ($\phi_i$), so the on-call engineer instantly knows whether the culprit is `dRTT/dt`, a MTU mismatch, or interface discards ([[TreeSHAP Explainability]]).
- **Counterfactual Simulator** allowing the operator to test QoS throttle policies before firing commands to the switch ([[Counterfactual What-If Simulation]]).

> [!quote] Student Engineer's Note
> They tell you that as an undergraduate, you can only write toy scripts and follow tutorials. NetPredict is proof that with first-principles physics, mathematical rigor, and unyielding persistence at 4 AM, you can build systems that rival enterprise infrastructure software.

Next, I wrote the complete operational runbook for anyone sitting in the operator's chair: see the [[Operator Guide - Using NetPredict]]. Onwards!
