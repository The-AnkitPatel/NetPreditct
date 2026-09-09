/*
 * NetPredict eBPF Kernel Probe: High-Precision Network Telemetry
 * Hooks into Linux Kernel TCP Stack for Zero-Overhead Telemetry
 * Targets: sock_ops RTT measurements, TCP retransmissions, and buffer drops.
 */

#include <uapi/linux/bpf.h>
#include <uapi/linux/tcp.h>
#include <linux/if_ether.h>
#include <linux/ip.h>

struct tcp_metrics_t {
    __u64 timestamp_ns;
    __u32 srtt_us;
    __u32 mdev_us;
    __u32 retrans_total;
    __u32 packets_out;
    __u32 lost_out;
};

// BPF Perf Event Output Map
BPF_PERF_OUTPUT(tcp_events);

// BPF Hash Table tracking connection state
BPF_HASH(conn_tracker, __u64, struct tcp_metrics_t);

// Hook 1: Trace TCP Retransmissions
int trace_tcp_retransmit_skb(struct pt_regs *ctx, struct sock *sk) {
    if (!sk) return 0;

    struct tcp_sock *tp = (struct tcp_sock *)sk;
    struct tcp_metrics_t metrics = {};

    metrics.timestamp_ns = bpf_ktime_get_ns();
    metrics.srtt_us = tp->srtt_us >> 3; // Smooth RTT in microseconds
    metrics.mdev_us = tp->mdev_us >> 1; // RTT variance (jitter)
    metrics.retrans_total = tp->total_retrans;
    metrics.lost_out = tp->lost_out;

    tcp_events.perf_submit(ctx, &metrics, sizeof(metrics));
    return 0;
}

// Hook 2: Socket State Callback for Latency Inflection
int bpf_sock_ops_cb(struct bpf_sock_ops *skops) {
    __u32 op = skops->op;

    if (op == BPF_SOCK_OPS_RTT_CB_FLAG || op == BPF_SOCK_OPS_RETRANS_CB) {
        struct tcp_metrics_t metrics = {};
        metrics.timestamp_ns = bpf_ktime_get_ns();
        metrics.srtt_us = skops->srtt >> 3;
        metrics.retrans_total = skops->total_retrans;

        // Trace event into ring buffer
        tcp_events.perf_submit(skops, &metrics, sizeof(metrics));
    }
    return 0;
}
