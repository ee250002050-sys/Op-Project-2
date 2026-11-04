"""Discrete-time packet-level WSN simulator (educational abstraction).

Models packet generation, hop-by-hop forwarding, probabilistic link delivery,
bounded retransmissions, radio TX/RX energy, idle listening and duty-cycled sleep.
It does not model a detailed MAC protocol, collisions, interference, fading,
carrier sensing, or hardware-specific timing.
"""
from __future__ import annotations
from collections import deque
import heapq
import math
import numpy as np


def _routes_to_sink(net, failed_edge=None):
    failed = tuple(sorted(failed_edge)) if failed_edge is not None else None
    outgoing = {}
    for u, v in net.arcs:
        if failed is not None and tuple(sorted((u, v))) == failed:
            continue
        outgoing.setdefault(u, []).append((v, float(net.tx_energy[(u, v)])))
    routes = {}
    for src in net.generation_rate:
        dist = {i: math.inf for i in range(len(net.positions))}
        prev = {}
        dist[src] = 0.0
        q = [(0.0, src)]
        while q:
            d, u = heapq.heappop(q)
            if d != dist[u]:
                continue
            if u == net.sink:
                break
            for v, w in outgoing.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(q, (nd, v))
        if math.isfinite(dist[net.sink]):
            path = [net.sink]
            cur = net.sink
            while cur != src:
                cur = prev[cur]
                path.append(cur)
            routes[src] = list(reversed(path))
        else:
            routes[src] = None
    return routes


def simulate_packets(net, *, duration_s=60.0, packet_bytes=128,
                     link_success=0.95, max_retries=2, idle_power_w=1e-4,
                     sleep_power_w=1e-6, awake_fraction=1.0,
                     time_step_s=0.1, seed=1, failed_edge=None):
    """Run packet-level simulation and return aggregate/node/link metrics.

    Each node can service one queued packet per time step. A failed transmission
    is retried up to max_retries additional times. Duty cycling is represented
    by a deterministic awake window within each one-second cycle.
    """
    if duration_s <= 0 or packet_bytes <= 0 or time_step_s <= 0:
        raise ValueError("Duration, packet size and time step must be positive.")
    if not 0 <= link_success <= 1 or not 0 < awake_fraction <= 1:
        raise ValueError("Link success must be 0..1 and awake fraction must be (0, 1].")
    if max_retries < 0:
        raise ValueError("Maximum retries cannot be negative.")
    rng = np.random.default_rng(int(seed))
    routes = _routes_to_sink(net, failed_edge)
    queues = {i: deque() for i in net.battery}
    generated_bits = {i: 0.0 for i in net.generation_rate}
    delivered = dropped = generated = tx_attempts = retries = 0
    tx_energy = {i: 0.0 for i in net.battery}
    rx_energy = {i: 0.0 for i in net.battery}
    idle_energy = {i: 0.0 for i in net.battery}
    sleep_energy = {i: 0.0 for i in net.battery}
    link_attempts = {}
    link_successes = {}
    packet_bits = int(packet_bytes) * 8
    steps = int(math.ceil(duration_s / time_step_s))

    for step in range(steps):
        now = step * time_step_s
        dt = min(time_step_s, duration_s - now)
        if dt <= 0:
            break
        # Listening/sleep energy is accrued for all battery-powered nodes.
        phase = now % 1.0
        awake = phase < awake_fraction
        for node in net.battery:
            idle_energy[node] += idle_power_w * dt * awake_fraction
            sleep_energy[node] += sleep_power_w * dt * (1.0 - awake_fraction)

        # Generate whole packets from each node's traffic accumulator.
        for src, rate in net.generation_rate.items():
            generated_bits[src] += float(rate) * dt
            while generated_bits[src] >= packet_bits:
                generated_bits[src] -= packet_bits
                generated += 1
                path = routes.get(src)
                if path is None:
                    dropped += 1
                else:
                    queues[src].append({"path": path, "hop": 0, "tries": 0})

        # One packet per awake node per slot; newly forwarded packets wait for
        # the next slot at their receiving node (store-and-forward abstraction).
        if awake:
            # Snapshot one packet per node before forwarding so a packet can
            # traverse at most one hop during a time step.
            ready = {node: (queues[node].popleft() if queues[node] else None)
                     for node in sorted(queues)}
            for node in sorted(queues):
                pkt = ready[node]
                if pkt is None:
                    continue
                path = pkt["path"]
                if pkt["hop"] >= len(path) - 1:
                    delivered += 1
                    continue
                nxt = path[pkt["hop"] + 1]
                arc = (node, nxt)
                if arc not in net.tx_energy:
                    dropped += 1
                    continue
                tx_attempts += 1
                link_attempts[arc] = link_attempts.get(arc, 0) + 1
                tx_energy[node] += float(net.tx_energy[arc]) * packet_bits
                success = bool(rng.random() < link_success)
                if success:
                    link_successes[arc] = link_successes.get(arc, 0) + 1
                    if nxt in rx_energy:
                        rx_energy[nxt] += float(net.rx_energy) * packet_bits
                    pkt["hop"] += 1
                    pkt["tries"] = 0
                    if nxt == net.sink:
                        delivered += 1
                    elif nxt in queues:
                        queues[nxt].append(pkt)
                    else:
                        dropped += 1
                else:
                    if pkt["tries"] < max_retries:
                        pkt["tries"] += 1
                        retries += 1
                        queues[node].appendleft(pkt)
                    else:
                        dropped += 1

    queued_at_end = sum(len(q) for q in queues.values())
    total_energy = {i: tx_energy[i] + rx_energy[i] + idle_energy[i] + sleep_energy[i]
                    for i in net.battery}
    remaining = {i: max(0.0, float(net.battery[i]) - total_energy[i]) for i in net.battery}
    depleted = [i for i, e in remaining.items() if e <= max(1e-12, float(net.battery[i]) * 1e-9)]
    link_rows = []
    for arc, attempts in sorted(link_attempts.items()):
        ok = link_successes.get(arc, 0)
        link_rows.append({"From": arc[0], "To": arc[1], "Attempts": attempts,
                          "Successful": ok, "Observed delivery (%)": 100.0 * ok / attempts})
    return {
        "duration_s": float(duration_s), "packet_bits": packet_bits,
        "generated_packets": generated, "delivered_packets": delivered,
        "dropped_packets": dropped, "delivery_ratio": delivered / generated if generated else 0.0,
        "tx_attempts": tx_attempts, "retries": retries,
        "throughput_bps": delivered * packet_bits / float(duration_s),
        "tx_energy": tx_energy, "rx_energy": rx_energy,
        "idle_energy": idle_energy, "sleep_energy": sleep_energy,
        "total_energy": total_energy, "remaining_energy": remaining,
        "depleted_nodes": depleted, "link_rows": link_rows,
        "routes": routes, "queued_at_end": queued_at_end,
        "failed_edge": tuple(sorted(failed_edge)) if failed_edge is not None else None,
    }
