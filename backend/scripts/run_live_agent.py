"""
NetPredict Real-Time Physical Interface Telemetry Daemon
Polls local OS network counters (psutil) and active socket latency,
streaming real-world hardware telemetry directly to the NetPredict API.
"""
import time
import argparse
import httpx
from backend.app.collectors.real_interface_collector import real_collector


def run_agent(endpoint: str, interval: float):
    print(f"[*] Starting NetPredict Live Telemetry Agent...")
    print(f"[*] Monitoring Host NIC: {real_collector.active_interface}")
    print(f"[*] Target Ingestion Endpoint: {endpoint}")
    print(f"[*] Polling Frequency: Every {interval}s")
    print("=" * 65)

    client = httpx.Client(timeout=3.0)
    samples_sent = 0

    try:
        while True:
            rec = real_collector.collect()
            payload = {
                "source": "live_physical_agent",
                "records": [rec.model_dump(mode="json")]
            }
            try:
                resp = client.post(endpoint, json=payload)
                samples_sent += 1
                print(
                    f"[{samples_sent:04d}] Sent: {rec.device_id}/{rec.interface_id} | "
                    f"RTT: {rec.rtt_ms}ms | Throughput: {rec.throughput_mbps}Mbps | "
                    f"CPU: {rec.cpu_util_pct}% | Loss: {rec.packet_loss_pct}% -> HTTP {resp.status_code}"
                )
            except Exception as e:
                print(f"[!] Ingestion connection failed: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[*] Live agent terminated by operator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetPredict Live Network Agent")
    parser.add_argument("--endpoint", default="http://localhost:8000/api/telemetry/ingest", help="Ingestion API endpoint")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    args = parser.parse_args()
    run_agent(args.endpoint, args.interval)
