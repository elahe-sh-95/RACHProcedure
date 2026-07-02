# -*- coding: utf-8 -*-
from datetime import datetime
import os
from scipy import signal, integrate, special
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
5G RACH Simulator for (critical) mMTC)

- Dynamic split NEW vs RETX (M_MAX=54)
- Patch1: p-persistent per class/pool
- Patch3: heavy-tailed adaptive backoff
- Group-Timed Dynamic Preamble Reservation (policy per slot)
- Clean line plots (success rate, collision prob, traffic, allocation, ECDFs)

This build (UE attempts cap = 10) adds:
  * Power-of-2 choices when picking a preamble (lower-loaded wins)
  * Controlled overflow from reserved -> general for burst UEs only (probabilistic)
  * Dynamic reservation with per-active-group caps (higher but safe)
  * Softer PI controller (lambda* = 0.45) for general pools
  * RESERVED pools driven a bit higher (TARGET_FILL≈0.78)

This edition adds:
  * Per-slot mean access success delay curve
  * Per-slot minimum preambles needed M_min^* to keep lambda <= lambda* (lambda*=0.45 by default)
  * Success rate and collision probability vs total devices curves
  * One-chart overlay of Delay / Success / Collision vs devices with emphasis on 120k

LowDelay patch:
  * More assertive PI and p-persistent on general pools
  * Softer backoff (especially for first collision)
  * Faster/shorter reservations and higher caps when bursty
  * Age-weighted attempt probabilities to pull in the tail
  * Greedier two-choice with a tiny local scan fallback
"""

# ============================
# Global knobs
# ============================

UNIFORM_PACKET_PROB = 1 / 15
SAVE_EPS = False

# ============================
# Plot style
# ============================

def setup_plot_style():
    plt.rcParams["figure.figsize"] = [8, 5]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlepad"] = 15
    plt.rcParams["axes.labelpad"] = 10
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.1
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

def save_high_quality_plot(filename):
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    if SAVE_EPS:
        orig = mpl.rcParams.get("legend.framealpha", None)
        try:
            mpl.rcParams["legend.framealpha"] = 1.0
            plt.savefig(f"{filename}.eps", format="eps", bbox_inches="tight")
        finally:
            if orig is None:
                del mpl.rcParams["legend.framealpha"]
            else:
                mpl.rcParams["legend.framealpha"] = orig

# ============================
# Plotting functions
# ============================

def plot_success_rate_comparison(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    success_rate = np.divide(
        per_slot_data["successfulUEsPerSlot"],
        per_slot_data["UEsPerSlot"],
        out=np.zeros_like(per_slot_data["successfulUEsPerSlot"], dtype=float),
        where=per_slot_data["UEsPerSlot"] > 0,
    )
    window_size = min(101, max(5, len(success_rate) // 10))
    if window_size % 2 == 0:
        window_size += 1
    smoothed_rate = np.convolve(success_rate, np.ones(window_size) / window_size, mode="same")
    fig, ax = plt.subplots()
    time_slots = np.arange(len(success_rate)) * 0.005
    ax.plot(time_slots, success_rate, alpha=0.5, label="Instantaneous", color="lightblue")
    ax.plot(time_slots, smoothed_rate, label=f"Moving Average (window={window_size})", color="blue")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Access Success Rate Over Time - {profile_name}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    mean_success = np.mean(success_rate) if success_rate.size else 0.0
    ax.axhline(y=mean_success, color="red", linestyle="--", label=f"Average: {mean_success:.3f}")
    ax.legend(loc="best")
    save_high_quality_plot(os.path.join(output_dir, f"success_rate_{profile_name}"))
    plt.close()

def plot_collision_probability(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    collision_prob = per_slot_data["collisionProbPerSlot"]
    window_size = min(101, max(5, len(collision_prob) // 10))
    if window_size % 2 == 0:
        window_size += 1
    smoothed_collision = np.convolve(collision_prob, np.ones(window_size) / window_size, mode="same")
    fig, ax = plt.subplots()
    time_slots = np.arange(len(collision_prob)) * 0.005
    ax.plot(time_slots, collision_prob, alpha=0.5, label="Instantaneous", color="lightcoral")
    ax.plot(time_slots, smoothed_collision, label=f"Moving Average (window={window_size})", color="red")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Collision Probability")
    ax.set_title(f"Collision Probability Over Time - {profile_name}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    mean_collision = np.mean(collision_prob) if collision_prob.size else 0.0
    ax.axhline(y=mean_collision, color="black", linestyle="--", label=f"Average: {mean_collision:.3f}")
    ax.legend(loc="best")
    save_high_quality_plot(os.path.join(output_dir, f"collision_prob_{profile_name}"))
    plt.close()

def plot_traffic_composition(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    fig, ax = plt.subplots()
    time_slots = np.arange(len(per_slot_data["newTraffic"])) * 0.005
    ax.plot(time_slots, per_slot_data["newTraffic"], label="New Traffic", color="green", linewidth=2)
    ax.plot(time_slots, per_slot_data["retxTraffic"], label="Retransmission Traffic", color="orange", linewidth=2)
    ax.plot(
        time_slots,
        per_slot_data["newTraffic"] + per_slot_data["retxTraffic"],
        label="Total Traffic",
        color="purple",
        linestyle="--",
        linewidth=2,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of UEs")
    ax.set_title(f"Traffic Composition Over Time - {profile_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    save_high_quality_plot(os.path.join(output_dir, f"traffic_composition_{profile_name}"))
    plt.close()

def plot_preamble_allocation(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    fig, ax = plt.subplots()
    time_slots = np.arange(len(per_slot_data["R_new_reserved"])) * 0.005
    total_reserved = per_slot_data["R_new_reserved"] + per_slot_data["R_retx_reserved"]
    total_general = per_slot_data["M_new_general"] + per_slot_data["M_retx_general"]
    ax.plot(time_slots, total_reserved, label="Reserved Preambles", color="royalblue", linewidth=2)
    ax.plot(time_slots, total_general, label="General Preambles", color="lightseagreen", linewidth=2)
    ax.plot(
        time_slots,
        total_reserved + total_general,
        label="Total Preambles",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of Preambles")
    ax.set_title(f"Preamble Allocation Over Time - {profile_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    save_high_quality_plot(os.path.join(output_dir, f"preamble_allocation_{profile_name}"))
    plt.close()

def plot_delay_ecdf(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    fig, ax = plt.subplots()
    if len(per_slot_data["success_delays_s"]) > 0:
        success_delays = np.sort(per_slot_data["success_delays_s"])
        success_ecdf = np.arange(1, len(success_delays) + 1) / len(success_delays)
        ax.plot(success_delays, success_ecdf, label="Success Delays", color="green", linewidth=2)
    if len(per_slot_data["dropped_delays_s"]) > 0:
        dropped_delays = np.sort(per_slot_data["dropped_delays_s"])
        dropped_ecdf = np.arange(1, len(dropped_delays) + 1) / len(dropped_delays)
        ax.plot(dropped_delays, dropped_ecdf, label="Dropped UEs Lifetime", color="red", linewidth=2)
    ax.set_xlabel("Delay (seconds)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Empirical CDF of Delays - {profile_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    save_high_quality_plot(os.path.join(output_dir, f"delay_ecdf_{profile_name}"))
    plt.close()

def plot_utilization_metrics(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    time_slots = np.arange(len(per_slot_data["usedPreambles"])) * 0.005
    utilization = per_slot_data["usedPreambles"] / 54.0
    ax1.plot(time_slots, utilization, label="Preamble Utilization", color="teal", linewidth=2)
    ax1.set_ylabel("Utilization Ratio")
    ax1.set_title(f"System Utilization Over Time - {profile_name}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    congestion_ratio = np.divide(
        per_slot_data["congestedPreambles"],
        per_slot_data["usedPreambles"],
        out=np.zeros_like(per_slot_data["congestedPreambles"], dtype=float),
        where=per_slot_data["usedPreambles"] > 0,
    )
    ax2.plot(time_slots, congestion_ratio, label="Congestion Ratio", color="coral", linewidth=2)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Congestion Ratio")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, f"utilization_metrics_{profile_name}"))
    plt.close()

# ===== New plots =====

def plot_mean_delay_per_slot(per_slot_data, profile_name, output_dir):
    setup_plot_style()
    y = per_slot_data["mean_success_delay_per_slot_s"]
    t = np.arange(len(y)) * 0.005
    if len(y) >= 51:
        w = 51 if len(y) < 201 else 101
        if w % 2 == 0:
            w += 1
        y_smooth = np.convolve(y, np.ones(w) / w, mode="same")
    else:
        y_smooth = y
    fig, ax = plt.subplots()
    ax.plot(t, y, alpha=0.35, label="Per-slot mean delay (s)")
    ax.plot(t, y_smooth, label="Moving Average", linewidth=2)
    valid_mask = per_slot_data["success_count_per_slot"] > 0
    valid = y[valid_mask]
    if valid.size:
        gmean = float(np.mean(valid))
        ax.axhline(gmean, linestyle="--", label=f"Global mean: {gmean:.4f}s")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean access success delay (s)")
    ax.set_title(f"Mean Access Success Delay per Slot - {profile_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    save_high_quality_plot(os.path.join(output_dir, f"mean_delay_per_slot_{profile_name}"))
    plt.close()

def plot_min_required_preambles(per_slot_data, profile_name, output_dir, lambda_star=0.45, M_MAX=54):
    setup_plot_style()
    attempts = per_slot_data["attemptedUEsPerSlot"].astype(float)
    used_now = per_slot_data["usedPreambles"].astype(float)
    M_star = np.ceil(attempts / max(lambda_star, 1e-9)).astype(int)
    M_star = np.clip(M_star, 1, M_MAX)
    t = np.arange(len(M_star)) * 0.005
    fig, ax = plt.subplots()
    ax.plot(t, M_star, label=r"$M_{\min}^{\ast}$ needed (per slot)", linewidth=2)
    ax.plot(t, used_now, label="Actual used preambles", linestyle="--", alpha=0.85)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Preambles (count)")
    ax.set_title(f"Minimum Preambles Needed vs. Actual Usage - {profile_name} (lambda*={lambda_star})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    save_high_quality_plot(os.path.join(output_dir, f"min_required_preambles_{profile_name}"))
    plt.close()

def plot_success_vs_devices(delay_data, output_dir):
    setup_plot_style()
    x = np.array(delay_data["device_counts"])
    y = np.array(delay_data["success_rates"])
    fig, ax = plt.subplots()
    ax.plot(x, y, "o--", linewidth=2, markersize=7)
    ax.set_xlabel("Total devices")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    total = delay_data.get("original_total", None)
    if total is not None:
        idx = np.where(x == total)[0]
        if idx.size:
            ax.scatter([x[idx[0]]], [y[idx[0]]], s=70)
    ax.set_title("Success rate vs total devices")
    save_high_quality_plot(os.path.join(output_dir, "success_rate_vs_devices"))
    plt.close()

def plot_collision_vs_devices(delay_data, output_dir):
    setup_plot_style()
    x = np.array(delay_data["device_counts"])
    y = np.array(delay_data["collision_probs"])
    fig, ax = plt.subplots()
    ax.plot(x, y, "s--", linewidth=2, markersize=7)
    ax.set_xlabel("Total devices")
    ax.set_ylabel("Collision probability")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    total = delay_data.get("original_total", None)
    if total is not None:
        idx = np.where(x == total)[0]
        if idx.size:
            ax.scatter([x[idx[0]]], [y[idx[0]]], s=70)
    ax.set_title("Collision probability vs total devices")
    save_high_quality_plot(os.path.join(output_dir, "collision_prob_vs_devices"))
    plt.close()

def plot_all_metrics_vs_devices_onechart(delay_data, output_dir):
    setup_plot_style()
    x = np.array(delay_data["device_counts"])
    delay = np.array(delay_data["avg_delays"])
    succ = np.array(delay_data["success_rates"])
    coll = np.array(delay_data["collision_probs"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Total devices")
    ax1.set_ylabel("Average access delay (s)")
    l1, = ax1.plot(x, delay, "o-", linewidth=2, markersize=7, label="Delay")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Rate / Probability (0-1)")
    l2, = ax2.plot(x, succ, "s--", linewidth=2, markersize=6, label="Success rate")
    l3, = ax2.plot(x, coll, "^:", linewidth=2, markersize=6, label="Collision prob.")
    ax2.set_ylim(0, 1.05)

    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Delay, Success rate, and Collision probability vs Total devices")
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, "all_metrics_vs_devices_onechart"))
    plt.close()

# ============================
# Distributions & Generators
# ============================

def betaDistribution(t, T=10.0):
    alpha = 3.0
    beta = 4.0
    betaFunction = special.beta(alpha, beta)
    pdf = (t ** (alpha - 1.0) * (T - t) ** (beta - 1.0)) / (T ** (alpha + beta - 1.0) * betaFunction)
    return pdf

def uniformTrafficIntensity(T, frameSize, numDevices, packetProbability=UNIFORM_PACKET_PROB):
    slots = int(T / frameSize)
    packetsPerSlotPerDev = packetProbability * frameSize
    randomSeries = np.random.uniform(0.0, 1.0, (slots, numDevices))
    uniformTraffic = np.sum(randomSeries <= packetsPerSlotPerDev, axis=1)
    return uniformTraffic.astype(int)

def burstTrafficIntensity(numDevices, frameSize, Tb=10.0):
    slots = int(Tb / frameSize)
    packetsPerSlotPerDev = np.zeros((slots, numDevices))
    tVector = np.linspace(0.0, Tb - frameSize, slots)
    for i, t in enumerate(tVector):
        val = integrate.quad(lambda x: betaDistribution(x, Tb), t, t + frameSize)[0]
        packetsPerSlotPerDev[i, :] = val
    randomSeries = np.random.uniform(0.0, 1.0, (slots, numDevices))
    burstTraffic = np.sum(randomSeries <= packetsPerSlotPerDev, axis=1)
    return burstTraffic.astype(int)

def eventGenerator(T, frameSize, eventProbability):
    slots = int(T / frameSize)
    eventProbabilityPerSlot = eventProbability * frameSize
    events = np.random.uniform(0.0, 1.0, slots) <= eventProbabilityPerSlot
    eventsTime = np.where(events)[0]
    numEvents = int(np.sum(events))
    Tb = np.random.uniform(8.0, 15.0, numEvents)  # seconds
    Tb = Tb - (Tb % frameSize)
    lastBurstEnd = 0
    overlapping = []
    for i, trig in enumerate(eventsTime):
        if lastBurstEnd > trig:
            overlapping.append(i)
        else:
            lastBurstEnd = trig + int(Tb[i] / frameSize)
    eventsTime = np.delete(eventsTime, overlapping)
    Tb = np.delete(Tb, overlapping)
    return eventsTime, Tb

def newArivals(numDevicesVec, eventProbabilities, T, frameSize):
    slots = int(T / frameSize)
    G = len(numDevicesVec)
    arrivals_total = np.zeros(slots, dtype=int)
    arrivals_per_group = np.zeros((slots, G), dtype=int)
    eventsAll = []
    TbsAll = []
    for i, numDevices in enumerate(numDevicesVec):
        groupTraffic = uniformTrafficIntensity(T, frameSize, numDevices)
        eventsTime, Tb = eventGenerator(T, frameSize, eventProbabilities[i])
        eventsAll.append(eventsTime)
        TbsAll.append(Tb)
        for k, event in enumerate(eventsTime):
            dur = int(Tb[k] / frameSize)
            end_idx = event + dur
            burstTraffic = burstTrafficIntensity(numDevices, frameSize, Tb[k])
            if end_idx >= slots:
                end_idx = slots
                lastPoint = end_idx - event
                groupTraffic[event:end_idx] = burstTraffic[:lastPoint]
            else:
                groupTraffic[event:end_idx] = burstTraffic
        arrivals_total += groupTraffic
        arrivals_per_group[:, i] = groupTraffic
    return arrivals_total, arrivals_per_group, eventsAll, TbsAll

def burst_mask_from_events(eventsAll, TbsAll, slots, frameSize):
    G = len(eventsAll)
    mask = np.zeros((slots, G), dtype=bool)
    for g in range(G):
        for k, start in enumerate(eventsAll[g]):
            dur_slots = int(TbsAll[g][k] / frameSize)
            end = min(slots, start + dur_slots)
            if start < end:
                mask[start:end, g] = True
    return mask

# ============================
# UE Class
# ============================

class UE:
    def __init__(self, group_id, first_slot=None):
        self.group = group_id
        self.transmissions = 0
        self.preamble = 0
        self.backoffCounter = 0
        self.first_slot = first_slot
        self.success = False

# ============================
# Dynamic Reservation Policy
# ============================

class DynamicReservationPolicy:
    def __init__(
        self,
        G,
        M_MAX=54,
        base_new=2,
        base_retx=2,
        max_per_group=6,
        hard_cap_total=16,
        cap_per_active=5,
        w_burst=1.6,
        w_retx_share=1.0,
        w_backlog=0.6,
        tau_on=0.55,
        tau_off=0.35,
        ramp_up=2,
        ramp_down=3,
        cooldown_slots=0,
        min_when_on=2,
    ):
        self.G = G
        self.M_MAX = M_MAX
        self.base_new = base_new
        self.base_retx = base_retx
        self.max_per_group = max_per_group
        self.hard_cap_total = min(hard_cap_total, M_MAX)
        self.cap_per_active = cap_per_active
        self.w_burst = w_burst
        self.w_retx_share = w_retx_share
        self.w_backlog = w_backlog
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.cooldown_slots = cooldown_slots
        self.min_when_on = min_when_on

        self.on_flags = np.zeros(G, dtype=bool)
        self.on_until = np.zeros(G, dtype=int)
        self.curr_new = np.zeros(G, dtype=int)
        self.curr_retx = np.zeros(G, dtype=int)

    def step(self, slot, burst_row, U_new_g, U_retx_g, back_new_g, back_retx_g):
        eps = 1e-9
        target_new = np.zeros(self.G, dtype=float)
        target_retx = np.zeros(self.G, dtype=float)

        active_groups = [
            g for g in range(self.G) if burst_row[g] or (back_new_g[g] + back_retx_g[g]) > 0
        ]
        n_active = len(active_groups)
        eff_cap = min(self.hard_cap_total, self.cap_per_active * n_active, self.M_MAX)

        for g in range(self.G):
            re_sum = U_new_g[g] + U_retx_g[g]
            retx_share = (U_retx_g[g] / (re_sum + eps)) if re_sum > 0 else 0.0
            backlog_present = (back_new_g[g] + back_retx_g[g]) > 0
            score = (
                self.w_burst * (1.0 if burst_row[g] else 0.0)
                + self.w_retx_share * retx_share
                + self.w_backlog * backlog_present
            )

            if self.on_flags[g]:
                if (score < self.tau_off) and (slot >= self.on_until[g]):
                    self.on_flags[g] = False
                else:
                    self.on_until[g] = max(self.on_until[g], slot + self.cooldown_slots)
            else:
                if (score >= self.tau_on) or burst_row[g]:
                    self.on_flags[g] = True
                    self.on_until[g] = slot + self.cooldown_slots

            if self.on_flags[g]:
                target_total = max(
                    self.min_when_on, self.base_new + self.base_retx + int(4 * score)
                )
                target_total = min(target_total, self.max_per_group)
                tn = int(round(target_total * (1.0 - retx_share)))
                tr = target_total - tn
                target_new[g] = max(0, tn)
                target_retx[g] = max(0, tr)
            else:
                target_new[g] = 0
                target_retx[g] = 0

        def ramp(curr, tgt):
            diff = tgt - curr
            if diff > 0:
                step = min(self.ramp_up, diff)
            else:
                step = max(-self.ramp_down, diff)
            return curr + step

        next_new = np.array(
            [ramp(self.curr_new[g], int(target_new[g])) for g in range(self.G)], dtype=int
        )
        next_retx = np.array(
            [ramp(self.curr_retx[g], int(target_retx[g])) for g in range(self.G)], dtype=int
        )

        total_req = int(next_new.sum() + next_retx.sum())
        budget = int(eff_cap)
        if (total_req > budget) and (total_req > 0):
            scale = budget / total_req
            next_new = np.floor(next_new * scale).astype(int)
            next_retx = np.floor(next_retx * scale).astype(int)
            leftover = budget - int(next_new.sum() + next_retx.sum())
            idx = 0
            order = [("n", i) for i in range(self.G)] + [("r", i) for i in range(self.G)]
            while leftover > 0 and order:
                tag, i = order[idx % len(order)]
                if tag == "n":
                    if next_new[i] < self.max_per_group:
                        next_new[i] += 1
                        leftover -= 1
                else:
                    if next_retx[i] < self.max_per_group:
                        next_retx[i] += 1
                        leftover -= 1
                idx += 1

        self.curr_new = next_new
        self.curr_retx = next_retx

        reserved_map = {}
        for g in range(self.G):
            if (self.curr_new[g] > 0) or (self.curr_retx[g] > 0):
                reserved_map[g] = {"new": int(self.curr_new[g]), "retx": int(self.curr_retx[g])}
        return reserved_map

# ============================
# Core Engine
# ============================

def actualTrafficPattern(
    arrivals_per_group,
    burst_mask,
    frameSize=0.005,
    backoffBool=True,
    PERSIST_K_GEN=0.48,
    PERSIST_K_RES=1.20,
    TARGET_FILL=0.78,
    SHORT_SKIP_MIN=1,
    SHORT_SKIP_MAX=5,
    BACKOFF_BASE_MS=40,
    RETX_PRESSURE_GAIN=3.5,
    BACKOFF_CAP_MS_MAX=550,
    MIN_RETX_PREAMBLES=0,
    STARVATION_SHARE=0.5,
    STARVATION_MULTIPLIER=2.0,
    OVERFLOW_P=0.35,
    RESERVATION=None,
    CARVE_OUT=True,
    RES_POLICY=None,
):
    M_MAX = 54
    preambleTransMax = 10
    slots, G = arrivals_per_group.shape
    eps = 1e-9

    UEsPerSlot = np.zeros(slots, dtype=int)
    successfulUEsPerSlot = np.zeros(slots, dtype=int)
    congestedPreambles = np.zeros(slots, dtype=int)
    freePreambles = np.zeros(slots, dtype=int)
    usedPreambles = np.zeros(slots, dtype=int)
    newTraffic = np.zeros(slots, dtype=int)
    retxTraffic = np.zeros(slots, dtype=int)
    M_new_series = np.zeros(slots, dtype=int)
    M_retx_series = np.zeros(slots, dtype=int)
    R_new_series = np.zeros(slots, dtype=int)
    R_retx_series = np.zeros(slots, dtype=int)

    collidedUEsPerSlot = np.zeros(slots, dtype=int)
    attemptedUEsPerSlot = np.zeros(slots, dtype=int)
    collisionProbPerSlot = np.zeros(slots, dtype=float)

    # NEW: preambles with exactly one user (successful preambles) per slot
    singleUserPreamblesPerSlot = np.zeros(slots, dtype=int)

    success_delay_sum_slots = np.zeros(slots, dtype=float)
    success_count_slots = np.zeros(slots, dtype=int)

    success_delays_slots = []
    dropped_delays_slots = []

    UEs = []
    last_state = np.zeros(M_MAX, dtype=int)

    # PI controller (more assertive)
    k_gen = 1.0
    e_int = 0.0
    KP = 0.25
    KI = 0.015
    KGEN_MIN, KGEN_MAX = 0.55, 1.60
    LAMBDA_TARGET = 0.55

    def scale_reserved(req_new, req_retx, budget):
        total_req = sum(req_new) + sum(req_retx)
        if total_req <= budget or total_req == 0:
            return req_new, req_retx
        scale = budget / total_req
        rn = [int(np.floor(x * scale)) for x in req_new]
        rr = [int(np.floor(x * scale)) for x in req_retx]
        used = sum(rn) + sum(rr)
        leftover = max(0, budget - used)
        idx = 0
        order = [("n", i) for i in range(len(rn))] + [("r", i) for i in range(len(rr))]
        while leftover > 0 and order:
            tag, i = order[idx % len(order)]
            if tag == "n":
                rn[i] += 1
            else:
                rr[i] += 1
            leftover -= 1
            idx += 1
        return rn, rr

    rng = np.random.default_rng()

    # Greedier two-choice with tiny fallback scan
    def two_choice_pick(r):
        a, b = r
        if b <= a:
            return None
        if b - a == 1:
            return a
        i = rng.integers(a, b)
        j = rng.integers(a, b)
        cand = i if preambleCounter[i] <= preambleCounter[j] else j
        # local fallback: peek a short window for a lighter slot
        if preambleCounter[cand] > 0:
            window = range(a, min(b, a + 6))
            cand2 = min(window, key=lambda x: preambleCounter[x])
            if preambleCounter[cand2] < preambleCounter[cand]:
                cand = cand2
        return cand

    # Age weighting to reduce long tails
    def age_weight(dev, slot, alpha=0.15):
        age = max(1, slot - dev.first_slot)
        return min(1.0, 1.0 - np.exp(-alpha * age))

    for slot in range(slots):
        # arrivals
        for g in range(G):
            for _ in range(int(arrivals_per_group[slot, g])):
                UEs.append(UE(group_id=g, first_slot=slot))

        # counters
        U_new_g = np.zeros(G, dtype=int)
        U_retx_g = np.zeros(G, dtype=int)
        back_new_g = np.zeros(G, dtype=int)
        back_retx_g = np.zeros(G, dtype=int)
        for dev in UEs:
            if dev.backoffCounter == 0:
                if dev.transmissions == 0:
                    U_new_g[dev.group] += 1
                else:
                    U_retx_g[dev.group] += 1
            else:
                if dev.transmissions == 0:
                    back_new_g[dev.group] += 1
                else:
                    back_retx_g[dev.group] += 1

        U_new_total = int(U_new_g.sum())
        U_retx_total = int(U_retx_g.sum())
        newTraffic[slot] = U_new_total
        retxTraffic[slot] = U_retx_total

        # reservation
        if RES_POLICY is not None:
            reserved_map = RES_POLICY.step(
                slot, burst_mask[slot], U_new_g, U_retx_g, back_new_g, back_retx_g
            )
        else:
            req_new, req_retx, active_groups = [], [], []
            for g in range(G):
                if burst_mask[slot, g]:
                    rconf = (RESERVATION or {}).get(g, {"new": 0, "retx": 0})
                    req_new.append(int(rconf.get("new", 0)))
                    req_retx.append(int(rconf.get("retx", 0)))
                    active_groups.append(g)
            if active_groups:
                rn, rr = scale_reserved(req_new, req_retx, M_MAX)
                reserved_map = {
                    g: {"new": rn[i], "retx": rr[i]} for i, g in enumerate(active_groups)
                }
            else:
                reserved_map = {}

        R_new = sum(v["new"] for v in reserved_map.values()) if reserved_map else 0
        R_retx = sum(v["retx"] for v in reserved_map.values()) if reserved_map else 0
        reserved_total = min(M_MAX, R_new + R_retx)
        R_new_series[slot] = R_new
        R_retx_series[slot] = R_retx

        M_general = max(0, M_MAX - reserved_total)

        nonburst_mask = ~burst_mask[slot]
        U_new_general = int(U_new_g[nonburst_mask].sum())
        U_retx_general = int(U_retx_g[nonburst_mask].sum())
        total_general_cont = U_new_general + U_retx_general

        if M_general == 0 or total_general_cont == 0:
            M_new_gen = 0
            M_retx_gen = 0 if M_general == 0 else M_general
        else:
            M_new_gen = int(np.floor(M_general * (U_new_general / (total_general_cont + eps))))
            M_new_gen = max(0, min(M_new_gen, M_general))
            M_retx_gen = M_general - M_new_gen

        # starvation guard for RETX in general
        if MIN_RETX_PREAMBLES > 0 and U_retx_general > 0:
            retx_share = (
                U_retx_general / (total_general_cont + eps) if total_general_cont > 0 else 0.0
            )
            retx_starved = (U_retx_general >= STARVATION_MULTIPLIER * max(1, M_retx_gen)) or (
                retx_share >= STARVATION_SHARE
            )
            if retx_starved:
                M_retx_gen = max(M_retx_gen, MIN_RETX_PREAMBLES)
                M_retx_gen = min(M_retx_gen, M_general)
                M_new_gen = max(0, M_general - M_retx_gen)

        M_new_series[slot] = M_new_gen
        M_retx_series[slot] = M_retx_gen

        # p-persistent
        p_new_g = np.zeros(G, dtype=float)
        p_retx_g = np.zeros(G, dtype=float)
        for g in range(G):
            if g in reserved_map and burst_mask[slot, g]:
                Rn = reserved_map[g]["new"]
                Rr = reserved_map[g]["retx"]
                if Rn > 0:
                    desired_attempts_new = TARGET_FILL * Rn
                    p_new_g[g] = (
                        1.0
                        if U_new_g[g] == 0
                        else min(1.0, 1.10 * desired_attempts_new / (U_new_g[g] + eps))
                    )
                else:
                    p_new_g[g] = 0.0
                if Rr > 0:
                    desired_attempts_retx = TARGET_FILL * Rr
                    p_retx_g[g] = (
                        1.0
                        if U_retx_g[g] == 0
                        else min(1.0, 1.10 * desired_attempts_retx / (U_retx_g[g] + eps))
                    )
                else:
                    p_retx_g[g] = 0.0
            else:
                p_new_g[g] = 0.0
                p_retx_g[g] = 0.0

        # more assertive general p
        PERSIST_K_GEN_eff = PERSIST_K_GEN * k_gen
        p_new_gen = (
            0.0
            if M_new_gen == 0
            else (
                1.0
                if U_new_general == 0
                else min(1.0, PERSIST_K_GEN_eff * M_new_gen / (U_new_general + eps))
            )
        )
        p_retx_gen = (
            0.0
            if M_retx_gen == 0
            else (
                1.0
                if U_retx_general == 0
                else min(1.0, PERSIST_K_GEN_eff * M_retx_gen / (U_retx_general + eps))
            )
        )

        preambleCounter = {k: 0 for k in range(M_MAX)}
        ranges = {}
        idx = 0
        active_groups = list(reserved_map.keys())

        for g in active_groups:
            n = reserved_map[g]["new"]
            if n > 0 and idx < M_MAX:
                a = idx
                b = min(M_MAX, idx + n)
                ranges[(g, "N")] = (a, b)
                idx = b
        for g in active_groups:
            n = reserved_map[g]["retx"]
            if n > 0 and idx < M_MAX:
                a = idx
                b = min(M_MAX, idx + n)
                ranges[(g, "R")] = (a, b)
                idx = b
        if M_new_gen > 0 and idx < M_MAX:
            a = idx
            b = min(M_MAX, idx + M_new_gen)
            ranges[("GEN", "N")] = (a, b)
            idx = b
        if M_retx_gen > 0 and idx < M_MAX:
            a = idx
            b = min(M_MAX, idx + M_retx_gen)
            ranges[("GEN", "R")] = (a, b)
            idx = b

        # attempt gating with age boost
        def allow_attempt_reserved(dev):
            base = p_new_g[dev.group] if dev.transmissions == 0 else p_retx_g[dev.group]
            return np.random.rand() < min(1.0, base * (0.75 + 0.5 * age_weight(dev, slot)))

        def allow_attempt_general(dev):
            base = p_new_gen if dev.transmissions == 0 else p_retx_gen
            return np.random.rand() < min(1.0, base * (0.75 + 0.5 * age_weight(dev, slot)))

        contenders_this_slot = 0
        for dev in UEs:
            if dev.backoffCounter > 0:
                dev.preamble = 99
                continue

            in_burst = burst_mask[slot, dev.group]
            took_action = False

            if in_burst and dev.group in reserved_map:
                if (
                    dev.transmissions == 0
                    and reserved_map[dev.group]["new"] > 0
                    and (dev.group, "N") in ranges
                ):
                    if allow_attempt_reserved(dev):
                        pr = two_choice_pick(ranges[(dev.group, "N")])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True
                elif (
                    dev.transmissions > 0
                    and reserved_map[dev.group]["retx"] > 0
                    and (dev.group, "R") in ranges
                ):
                    if allow_attempt_reserved(dev):
                        pr = two_choice_pick(ranges[(dev.group, "R")])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True

                if not took_action:
                    go_overflow = (("GEN", "N") in ranges and dev.transmissions == 0) or (
                        ("GEN", "R") in ranges and dev.transmissions > 0
                    )
                    if go_overflow and (np.random.rand() < OVERFLOW_P):
                        pass
                    else:
                        skip_slots = np.random.randint(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                        dev.backoffCounter = skip_slots
                        dev.preamble = 99
                        continue

            if not took_action:
                if dev.transmissions == 0 and ("GEN", "N") in ranges and M_new_gen > 0:
                    if allow_attempt_general(dev):
                        pr = two_choice_pick(ranges[("GEN", "N")])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True
                elif dev.transmissions > 0 and ("GEN", "R") in ranges and M_retx_gen > 0:
                    if allow_attempt_general(dev):
                        pr = two_choice_pick(ranges[("GEN", "R")])
                        if pr is not None:
                            dev.preamble = pr
                            preambleCounter[pr] += 1
                            dev.transmissions += 1
                            took_action = True

                if not took_action:
                    skip_slots = np.random.randint(SHORT_SKIP_MIN, SHORT_SKIP_MAX + 1)
                    dev.backoffCounter = skip_slots
                    dev.preamble = 99
                    continue

            if dev.preamble != 99:
                contenders_this_slot += 1

        # outcomes per preamble
        attempts_this_slot = 0
        collided_UEs_this_slot = 0
        unused_pream = 0
        coll_pream = 0
        single_pream = 0  # NEW: count of c == 1 for this slot
        for j in range(M_MAX):
            c = preambleCounter[j]
            attempts_this_slot += c
            if c == 0:
                unused_pream += 1
                last_state[j] = 0
            elif c == 1:
                last_state[j] = 1
                single_pream += 1  # one-user preamble -> successful preamble
            else:
                coll_pream += 1
                last_state[j] = 2
                collided_UEs_this_slot += c

        # write per-slot counters
        singleUserPreamblesPerSlot[slot] = single_pream  # NEW
        congestedPreambles[slot] = coll_pream
        freePreambles[slot] = unused_pream
        usedPreambles[slot] = M_MAX - unused_pream

        attemptedUEsPerSlot[slot] = attempts_this_slot
        collidedUEsPerSlot[slot] = collided_UEs_this_slot
        collisionProbPerSlot[slot] = (
            collided_UEs_this_slot / attempts_this_slot if attempts_this_slot > 0 else 0.0
        )

        # PI update
        used_now = max(1, usedPreambles[slot])
        lambda_hat = attempts_this_slot / float(used_now)
        err = LAMBDA_TARGET - lambda_hat
        e_int = np.clip(e_int + err, -5.0, 5.0)
        k_gen = np.clip(k_gen + KP * err + KI * e_int, KGEN_MIN, KGEN_MAX)

        # success/backoff updates
        finishedUEs = []
        for dev in UEs:
            if dev.backoffCounter == 0:
                if dev.preamble != 99 and preambleCounter.get(dev.preamble, 0) == 1:
                    successfulUEsPerSlot[slot] += 1
                    d_slots = slot - dev.first_slot
                    success_delays_slots.append(d_slots)
                    success_delay_sum_slots[slot] += d_slots
                    success_count_slots[slot] += 1
                    finishedUEs.append(dev)
                elif dev.preamble != 99 and dev.transmissions >= preambleTransMax:
                    dropped_delays_slots.append(slot - dev.first_slot)
                    finishedUEs.append(dev)
                elif dev.preamble != 99:
                    if backoffBool:
                        if dev.transmissions == 1:
                            randomBackoff_ms = np.random.randint(0, 15)
                        else:
                            U_total_now = U_new_total + U_retx_total
                            pressure = (
                                U_retx_total / (U_total_now + eps) if U_total_now > 0 else 0.0
                            )
                            cap_ms = int(
                                min(
                                    BACKOFF_CAP_MS_MAX,
                                    BACKOFF_BASE_MS * (1 + RETX_PRESSURE_GAIN * pressure),
                                )
                            )
                            r = max(1e-9, 1 - np.random.rand())
                            randomBackoff_ms = int(cap_ms * (-np.log(r) / 2.0))
                    else:
                        randomBackoff_ms = 0
                    slots_backoff = int(np.ceil(randomBackoff_ms / (frameSize * 1000.0)))
                    dev.backoffCounter = slots_backoff
            else:
                dev.backoffCounter -= 1

        for dev in finishedUEs:
            UEs.remove(dev)
            del dev

        UEsPerSlot[slot] = contenders_this_slot

    total_successes = int(np.sum(successfulUEsPerSlot))
    total_contenders = int(np.sum(UEsPerSlot))
    success_rate_per_attempt = (
        total_successes / total_contenders if total_contenders > 0 else 0.0
    )

    success_delays_sec = np.array(success_delays_slots, dtype=float) * frameSize
    dropped_delays_sec = np.array(dropped_delays_slots, dtype=float) * frameSize

    overall_collision_probability = (
        float(np.sum(collidedUEsPerSlot)) / float(np.sum(attemptedUEsPerSlot))
        if np.sum(attemptedUEsPerSlot) > 0
        else 0.0
    )
    avg_slot_collision_probability = (
        float(np.mean(collisionProbPerSlot)) if collisionProbPerSlot.size else 0.0
    )

    mean_success_delay_per_slot = np.divide(
        success_delay_sum_slots * frameSize,
        success_count_slots,
        out=np.zeros_like(success_delay_sum_slots, dtype=float),
        where=success_count_slots > 0,
    )

    metrics = {
        "success_rate_per_attempt": success_rate_per_attempt,
        "total_successes": total_successes,
        "total_contenders": total_contenders,
        "avg_preamble_occupancy": float(np.mean(usedPreambles / float(M_MAX))) if M_MAX > 0 else 0.0,
        "avg_collided_preambles": float(np.mean(congestedPreambles)),
        "avg_free_preambles": float(np.mean(freePreambles)),
        "delay_stats": {
            "count_success": int(success_delays_sec.size),
            "avg_success_delay_s": float(np.mean(success_delays_sec)) if success_delays_sec.size else None,
            "p50_success_delay_s": float(np.percentile(success_delays_sec, 50)) if success_delays_sec.size else None,
            "p95_success_delay_s": float(np.percentile(success_delays_sec, 95)) if success_delays_sec.size else None,
            "count_dropped": int(dropped_delays_sec.size),
            "avg_dropped_lifetime_s": float(np.mean(dropped_delays_sec)) if dropped_delays_sec.size else None,
            "p95_dropped_lifetime_s": float(np.percentile(dropped_delays_sec, 95)) if dropped_delays_sec.size else None,
        },
        "overall_collision_probability": overall_collision_probability,
        "avg_slot_collision_probability": avg_slot_collision_probability,
    }

    per_slot = {
        "successfulUEsPerSlot": successfulUEsPerSlot,
        "UEsPerSlot": UEsPerSlot,
        "congestedPreambles": congestedPreambles,
        "freePreambles": freePreambles,
        "usedPreambles": usedPreambles,
        "newTraffic": newTraffic,
        "retxTraffic": retxTraffic,
        "M_new_general": M_new_series,
        "M_retx_general": M_retx_series,
        "R_new_reserved": R_new_series,
        "R_retx_reserved": R_retx_series,
        "success_delays_s": success_delays_sec,
        "dropped_delays_s": dropped_delays_sec,
        "collidedUEsPerSlot": collidedUEsPerSlot,
        "attemptedUEsPerSlot": attemptedUEsPerSlot,
        "collisionProbPerSlot": collisionProbPerSlot,
        "mean_success_delay_per_slot_s": mean_success_delay_per_slot,
        "success_count_per_slot": success_count_slots,
        # NEW: number of preambles with exactly one user (successful preambles) per slot
        "singleUserPreamblesPerSlot": singleUserPreamblesPerSlot,
    }

    return metrics, per_slot

# ============================
# Delay/Scalability vs devices
# ============================

def plot_delay_vs_devices(numDevicesVec, eventProbabilities, T, frameSize, output_dir, num_points=6):
    setup_plot_style()

    total_devices = sum(numDevicesVec)
    print(f"Total number of devices: {total_devices:,}")

    device_counts = np.array([60000, 90000, 120000, 150000, 180000], dtype=int)
    print(f"Testing device counts: {[f'{x:,}' for x in device_counts]}")

    avg_delays = []
    success_rates = []
    collision_probs = []

    print("Delay sensitivity")

    G = len(numDevicesVec)
    analysis_policy = DynamicReservationPolicy(
        G=G,
        M_MAX=54,
        base_new=2,
        base_retx=2,
        max_per_group=6,
        hard_cap_total=20,
        cap_per_active=6,
        w_burst=1.6,
        w_retx_share=1.0,
        w_backlog=0.6,
        tau_on=0.50,
        tau_off=0.40,
        ramp_up=3,
        ramp_down=4,
        cooldown_slots=int(0.25 / frameSize),
        min_when_on=2,
    )

    for device_count in tqdm(device_counts, desc="Simulation for different devices"):
        scale_factor = device_count / total_devices
        scaled_numDevicesVec = [max(1000, int(count * scale_factor)) for count in numDevicesVec]
        current_total = sum(scaled_numDevicesVec)
        if current_total != device_count:
            diff = device_count - current_total
            max_group_idx = np.argmax(scaled_numDevicesVec)
            scaled_numDevicesVec[max_group_idx] += diff

        arrivals_total, arrivals_per_group, eventsAll, TbsAll = newArivals(
            scaled_numDevicesVec, eventProbabilities, T, frameSize
        )
        slots = int(T / frameSize)
        burst_mask = burst_mask_from_events(eventsAll, TbsAll, slots, frameSize)

        analysis_policy.on_flags[:] = False
        analysis_policy.on_until[:] = 0
        analysis_policy.curr_new[:] = 0
        analysis_policy.curr_retx[:] = 0

        metrics, per_slot = actualTrafficPattern(
            arrivals_per_group,
            burst_mask,
            frameSize=frameSize,
            backoffBool=True,
            PERSIST_K_GEN=0.60,
            PERSIST_K_RES=1.10,
            TARGET_FILL=0.72,
            SHORT_SKIP_MIN=0,
            SHORT_SKIP_MAX=2,
            BACKOFF_BASE_MS=25,
            RETX_PRESSURE_GAIN=2.2,
            BACKOFF_CAP_MS_MAX=350,
            MIN_RETX_PREAMBLES=3,
            STARVATION_SHARE=0.45,
            STARVATION_MULTIPLIER=1.6,
            OVERFLOW_P=0.70,
            RESERVATION=None,
            CARVE_OUT=True,
            RES_POLICY=analysis_policy,
        )

        if metrics["delay_stats"]["avg_success_delay_s"] is not None:
            avg_delays.append(metrics["delay_stats"]["avg_success_delay_s"])
            success_rates.append(metrics["success_rate_per_attempt"])
            collision_probs.append(metrics["overall_collision_probability"])
        else:
            avg_delays.append(0.0)
            success_rates.append(0.0)
            collision_probs.append(0.0)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color1 = "tab:blue"
    ax1.set_xlabel("Total devices", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Average delay of success (s)", color=color1, fontsize=12)
    line1 = ax1.plot(
        device_counts,
        avg_delays,
        "o-",
        color=color1,
        linewidth=3,
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Average delay",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(device_counts)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Success rate", color=color2, fontsize=12)
    line2 = ax2.plot(
        device_counts, success_rates, "s--", color=color2, linewidth=2, markersize=8, label="Success rate"
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    color3 = "tab:green"
    line3 = ax1.plot(
        device_counts, collision_probs, "^:", color=color3, linewidth=2, markersize=8, label="Collision probability"
    )

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=10)
    plt.title("Effect of number of devices on RACH", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, "delay_vs_devices_comprehensive"))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        device_counts, avg_delays, "o-", color="blue", linewidth=3, markersize=10, markerfacecolor="white", markeredgewidth=2
    )
    ax.set_xlabel("Total Devices", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average access delay (s)", fontsize=12)
    ax.set_title("Effect of number of devices on access delay", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(device_counts)
    plt.tight_layout()
    save_high_quality_plot(os.path.join(output_dir, "delay_vs_devices_simple"))
    plt.close()

    delay_data = {
        "device_counts": device_counts.tolist(),
        "avg_delays": avg_delays,
        "success_rates": success_rates,
        "collision_probs": collision_probs,
        "original_total": total_devices,
    }
    torch.save(delay_data, os.path.join(output_dir, "delay_analysis_data.pt"))
    return delay_data

def plot_scalability_analysis(delay_data, output_dir):
    setup_plot_style()
    device_counts = np.array(delay_data["device_counts"])
    avg_delays = np.array(delay_data["avg_delays"])
    if len(device_counts) > 1 and avg_delays.size:
        delays_derivative = np.gradient(avg_delays, device_counts)
        threshold = np.mean(delays_derivative) + 2 * np.std(delays_derivative)
        breakpoint_idx = np.where(delays_derivative > threshold)[0]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        color1 = "tab:blue"
        ax1.plot(
            device_counts, avg_delays, "o-", color=color1, linewidth=3, markersize=8, label="Average delay"
        )
        ax1.set_xlabel("Number of devices")
        ax1.set_ylabel("Delay (s)", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax1_twin = ax1.twinx()
        color2 = "tab:red"
        ax1_twin.plot(
            device_counts, delays_derivative, "s--", color=color2, linewidth=2, markersize=6, label="Delay slope"
        )
        ax1_twin.set_ylabel("Delay slope", color=color2)
        ax1_twin.tick_params(axis="y", labelcolor=color2)

        if len(breakpoint_idx) > 0:
            bp_idx = breakpoint_idx[0]
            ax1.axvline(
                x=device_counts[bp_idx],
                color="red",
                linestyle=":",
                alpha=0.7,
                label=f"break point: {device_counts[bp_idx]:,} devices",
            )

        ax1.legend(loc="upper left")
        ax1_twin.legend(loc="upper right")

        normalized_delay = avg_delays / avg_delays[0] if avg_delays[0] > 0 else avg_delays
        normalized_devices = device_counts / device_counts[0]
        scalability = np.divide(
            normalized_devices, normalized_delay, out=np.zeros_like(normalized_devices, dtype=float), where=normalized_delay > 0
        )
        ax2.plot(device_counts, scalability, "^-", color="green", linewidth=2, markersize=8, label="Scalability factor")
        ax2.set_xlabel("Number of devices")
        ax2.set_ylabel("Scalability factor")
        ax2.legend()
        plt.suptitle("System scalability - RACH", fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_high_quality_plot(os.path.join(output_dir, "scalability_analysis"))
        plt.close()

# ============================
# Main
# ============================

if __name__ == "__main__":
    # Traffic Config
    numDevicesVec = [20000, 10000, 8000, 10000, 10000, 20000, 10000, 12000, 10000, 10000]  # sum = 120,000
    eventProbabilities = [0.006, 0.009, 0.09, 0.1, 0.2, 0.004, 0.004, 0.05, 0.1, 0.2]
    totalStreams = 1
    T = 10
    frameSize = 0.005
    slots = int(T / frameSize)
    G = len(numDevicesVec)

    # Dynamic Group-time Reservation Policy (LowDelay defaults)
    RES_POLICY = DynamicReservationPolicy(
        G=G,
        M_MAX=54,
        base_new=2,
        base_retx=2,
        max_per_group=6,
        hard_cap_total=20,
        cap_per_active=6,
        w_burst=1.6,
        w_retx_share=1.0,
        w_backlog=0.6,
        tau_on=0.50,
        tau_off=0.40,
        ramp_up=3,
        ramp_down=4,
        cooldown_slots=int(0.25 / frameSize),
        min_when_on=2,
    )

    # Profiles (RescuePlus kept for A/B; LowDelay is new)
    PARAM_PROFILES = [
        {
            "name": "LowDelay",
            "desc": "Aggressive general p, softer backoff, fast reservations",
            "PERSIST_K_GEN": 0.60,
            "PERSIST_K_RES": 1.10,
            "TARGET_FILL": 0.72,
            "SHORT_SKIP_MIN": 0,
            "SHORT_SKIP_MAX": 2,
            "BACKOFF_BASE_MS": 25,
            "RETX_PRESSURE_GAIN": 2.2,
            "BACKOFF_CAP_MS_MAX": 350,
            "MIN_RETX_PREAMBLES": 3,
            "STARVATION_SHARE": 0.45,
            "STARVATION_MULTIPLIER": 1.6,
            "OVERFLOW_P": 0.70,
        },
        {
            "name": "RescuePlus",
            "desc": "Higher burst reservation + controlled overflow + conservative PI",
            "PERSIST_K_GEN": 0.48,
            "PERSIST_K_RES": 1.20,
            "TARGET_FILL": 0.78,
            "SHORT_SKIP_MIN": 1,
            "SHORT_SKIP_MAX": 5,
            "BACKOFF_BASE_MS": 40,
            "RETX_PRESSURE_GAIN": 3.5,
            "BACKOFF_CAP_MS_MAX": 550,
            "MIN_RETX_PREAMBLES": 0,
            "STARVATION_SHARE": 0.5,
            "STARVATION_MULTIPLIER": 2.0,
            "OVERFLOW_P": 0.35,
        },
    ]

    # Storage
    TT = datetime.now()
    root_dir = r"D:\YAZD\Masters Thesis\Code\P-persistant and adaptive backoff\With tuned parameters\Tuned-V2\with collision probability\Result"
    if not os.path.isdir(root_dir):
        try:
            os.makedirs(root_dir, exist_ok=True)
        except Exception:
            root_dir = os.path.join(os.getcwd(), "A_generatedTraffic")
            os.makedirs(root_dir, exist_ok=True)

    # Run
    for prof in PARAM_PROFILES[:1]:  # run LowDelay by default; change to [:] to run both
        print(f"\n===== Running profile: {prof['name']}  ({prof['desc']}) =====")

        address1 = (
            root_dir
            + f"/{TT.strftime('%j')}_{TT.strftime('%a')}_"
            + f"{TT.strftime('%b')}{TT.strftime('%d')}_"
            + f"{TT.strftime('%H')}{TT.strftime('%M')}"
            + f"_Samples({totalStreams})({T}Sec)_{prof['name']}"
        )
        fileType = ".pt"
        address = address1 + fileType
        cnt = 1
        while os.path.isfile(address):
            address = address1 + f"({cnt})" + fileType
            cnt += 1

        plots_dir = os.path.join(root_dir, f"plots_{prof['name']}_{TT.strftime('%H%M')}")
        os.makedirs(plots_dir, exist_ok=True)

        Intensity = []
        Pattern = []
        aggregate_success = 0
        aggregate_contenders = 0
        aggregate_delays_success = []
        aggregate_delays_dropped = []
        aggregate_collided = 0
        aggregate_attempted = 0

        starttime = datetime.now()
        for i in tqdm(
            range(totalStreams), desc=f"Generating & simulating [{prof['name']}]", position=0, colour="red"
        ):
            arrivals_total, arrivals_per_group, eventsAll, TbsAll = newArivals(
                numDevicesVec, eventProbabilities, T, frameSize
            )
            traffic_smoothed = signal.savgol_filter(arrivals_total, 97, 2)
            burst_mask = burst_mask_from_events(eventsAll, TbsAll, slots, frameSize)

            RES_POLICY.on_flags[:] = False
            RES_POLICY.on_until[:] = 0
            RES_POLICY.curr_new[:] = 0
            RES_POLICY.curr_retx[:] = 0

            metrics, per_slot = actualTrafficPattern(
                arrivals_per_group,
                burst_mask,
                frameSize=frameSize,
                backoffBool=True,
                PERSIST_K_GEN=prof["PERSIST_K_GEN"],
                PERSIST_K_RES=prof["PERSIST_K_RES"],
                TARGET_FILL=prof["TARGET_FILL"],
                SHORT_SKIP_MIN=prof["SHORT_SKIP_MIN"],
                SHORT_SKIP_MAX=prof["SHORT_SKIP_MAX"],
                BACKOFF_BASE_MS=prof["BACKOFF_BASE_MS"],
                RETX_PRESSURE_GAIN=prof["RETX_PRESSURE_GAIN"],
                BACKOFF_CAP_MS_MAX=prof["BACKOFF_CAP_MS_MAX"],
                MIN_RETX_PREAMBLES=prof["MIN_RETX_PREAMBLES"],
                STARVATION_SHARE=prof["STARVATION_SHARE"],
                STARVATION_MULTIPLIER=prof["STARVATION_MULTIPLIER"],
                OVERFLOW_P=prof["OVERFLOW_P"],
                RESERVATION=None,
                CARVE_OUT=True,
                RES_POLICY=RES_POLICY,
            )

            Intensity.append([arrivals_total, eventsAll, TbsAll, traffic_smoothed])
            Pattern.append(per_slot)

            aggregate_success += metrics["total_successes"]
            aggregate_contenders += metrics["total_contenders"]
            aggregate_collided += int(np.sum(per_slot["collidedUEsPerSlot"]))
            aggregate_attempted += int(np.sum(per_slot["attemptedUEsPerSlot"]))
            if per_slot["success_delays_s"].size:
                aggregate_delays_success.append(per_slot["success_delays_s"])
            if per_slot["dropped_delays_s"].size:
                aggregate_delays_dropped.append(per_slot["dropped_delays_s"])

        agg_success_delays = (
            np.concatenate(aggregate_delays_success) if aggregate_delays_success else np.array([])
        )
        agg_dropped_delays = (
            np.concatenate(aggregate_delays_dropped) if aggregate_delays_dropped else np.array([])
        )

        overall_success_rate = (
            aggregate_success / aggregate_contenders if aggregate_contenders > 0 else 0.0
        )
        overall_collision_prob = (
            aggregate_collided / aggregate_attempted if aggregate_attempted > 0 else 0.0
        )
        agg_delay_stats = {
            "count_success": int(agg_success_delays.size),
            "avg_success_delay_s": float(np.mean(agg_success_delays)) if agg_success_delays.size else None,
            "p50_success_delay_s": float(np.percentile(agg_success_delays, 50)) if agg_success_delays.size else None,
            "p95_success_delay_s": float(np.percentile(agg_success_delays, 95)) if agg_success_delays.size else None,
            "count_dropped": int(agg_dropped_delays.size),
            "avg_dropped_lifetime_s": float(np.mean(agg_dropped_delays)) if agg_dropped_delays.size else None,
            "p95 dropped_lifetime_s": float(np.percentile(agg_dropped_delays, 95)) if agg_dropped_delays.size else None,
        }

        torch.save(
            {
                "Description": f"Dynamic split + Patch1/3 + Dynamic Group Reservation + PI control + 2-choice + controlled overflow [{prof['name']}]",
                "profile": prof,
                "numDevicesVec": numDevicesVec,
                "eventProbabilities": eventProbabilities,
                "T": T,
                "frameSize": frameSize,
                "Intensity": Intensity,
                "Pattern": Pattern,
                "overall_success_rate": overall_success_rate,
                "aggregate_success": aggregate_success,
                "aggregate_contenders": aggregate_contenders,
                "aggregate_delay_stats": agg_delay_stats,
                "overall_collision_probability": overall_collision_prob,
                "reservation_policy": {
                    "hard_cap_total": RES_POLICY.hard_cap_total,
                    "max_per_group": RES_POLICY.max_per_group,
                    "cap_per_active": RES_POLICY.cap_per_active,
                    "tau_on": RES_POLICY.tau_on,
                    "tau_off": RES_POLICY.tau_off,
                    "cooldown_slots": RES_POLICY.cooldown_slots,
                    "ramp_up": RES_POLICY.ramp_up,
                    "ramp_down": RES_POLICY.ramp_down,
                },
            },
            address,
        )

        print("\n==== Summary ==== ")
        print(f"Output file: {address}")
        print(f"Total contenders (attempts): {aggregate_contenders:,}")
        print(f"Total successes:             {aggregate_success:,}")
        print(f"Overall success rate:        {overall_success_rate:.4f}")
        print(f"Overall collision probability: {overall_collision_prob:.4f}")
        if agg_success_delays.size:
            print(f"Avg success delay (s):       {agg_delay_stats['avg_success_delay_s']:.4f}")
            print(f"P50 success delay (s):       {agg_delay_stats['p50_success_delay_s']:.4f}")
            print(f"P95 success delay (s):       {agg_delay_stats['p95_success_delay_s']:.4f}")
        else:
            print("No successful transmissions to report delay stats.")
        if agg_dropped_delays.size:
            print(f"Dropped UEs:                 {agg_delay_stats['count_dropped']:,}")
            print(f"Avg dropped lifetime (s):    {agg_delay_stats['avg_dropped_lifetime_s']:.4f}")
            print(f"P95 dropped lifetime (s):    {agg_delay_stats['p95 dropped_lifetime_s']:.4f}")
        else:
            print("No dropped UEs.")
        print("Wall-clock:", datetime.now() - starttime)

        # Plots (stream #0)
        print(f"\nGenerating publication-quality plots for {prof['name']}...")
        per_slot_0 = Pattern[0]

        plot_success_rate_comparison(per_slot_0, prof["name"], plots_dir)
        plot_collision_probability(per_slot_0, prof["name"], plots_dir)
        plot_traffic_composition(per_slot_0, prof["name"], plots_dir)
        plot_preamble_allocation(per_slot_0, prof["name"], plots_dir)
        plot_delay_ecdf(per_slot_0, prof["name"], plots_dir)
        plot_utilization_metrics(per_slot_0, prof["name"], plots_dir)

        # New plots
        plot_mean_delay_per_slot(per_slot_0, prof["name"], plots_dir)
        plot_min_required_preambles(per_slot_0, prof["name"], plots_dir)

        # Optional: sensitivity (kept as before, can be skipped for speed)
        if totalStreams <= 10:
            delay_data = plot_delay_vs_devices(
                numDevicesVec, eventProbabilities, T, frameSize, plots_dir, num_points=6
            )
            plot_scalability_analysis(delay_data, plots_dir)
            plot_success_vs_devices(delay_data, plots_dir)
            plot_collision_vs_devices(delay_data, plots_dir)
            plot_all_metrics_vs_devices_onechart(delay_data, plots_dir)

        print(f"All plots for {prof['name']} saved in high-quality formats in: {plots_dir}")
