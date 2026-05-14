import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

MACHINES = ["CARRIER-CHILLER-01", "CARRIER-VRF-UNIT-01"]
SAMPLES_PER_MACHINE = 10_000
DT = 0.1  # seconds between samples

# ─── Normal signal generators ─────────────────────────────────────────────────

def normal_signals(n, t):
    """Four correlated HVAC streams under normal thermodynamic harmony."""
    demand = (
        0.4 * np.sin(2 * np.pi * 0.02 * t)
        + 0.15 * np.sin(2 * np.pi * 0.007 * t)
        + np.random.normal(0, 0.05, n)
    )
    comp_power = np.clip(
        3.5
        + demand
        + np.random.normal(0, 0.08, n)
        + 0.02 * np.cumsum(np.random.normal(0, 0.004, n)),
        2.0, 6.0,
    )
    disc_pressure = 70.0 * comp_power + np.random.normal(0, 4.0, n)
    fan_rpm       = 340.0 * comp_power + np.random.normal(0, 30.0, n)
    supply_temp   = 18.0 - 2.0 * comp_power + np.random.normal(0, 0.3, n)
    return comp_power, disc_pressure, fan_rpm, supply_temp


# ─── Fault injectors ──────────────────────────────────────────────────────────

def inject_refrigerant_leak(comp_power, disc_pressure, fan_rpm, supply_temp, start, dur):
    """
    Refrigerant Leak — gas escapes, pressure falls + temp rises abruptly.
    Compressor stays HIGH (working hard); fan unaffected.
    """
    psi_drop  = np.random.uniform(0.30, 0.45) * disc_pressure[start:start + dur].mean()
    temp_rise = np.random.uniform(5.0, 9.0)
    disc_pressure[start:start + dur] -= psi_drop
    supply_temp[start:start + dur]   += temp_rise
    return comp_power, disc_pressure, fan_rpm, supply_temp


def inject_fan_failure(comp_power, disc_pressure, fan_rpm, supply_temp, start, dur):
    """
    Condenser Fan Failure — motor degrades, heat dissipation collapses.
    Fan RPM drops abruptly; everything else drifts upward gradually.
    """
    ramp     = np.linspace(0, 1, dur)
    rpm_drop = np.random.uniform(0.70, 0.90) * fan_rpm[start:start + dur].mean()
    fan_rpm[start:start + dur]        -= rpm_drop
    comp_power[start:start + dur]     += ramp * np.random.uniform(1.0, 1.8)
    disc_pressure[start:start + dur]  += ramp * np.random.uniform(40.0, 70.0)
    supply_temp[start:start + dur]    += ramp * np.random.uniform(3.0, 6.0)
    return comp_power, disc_pressure, fan_rpm, supply_temp


def inject_compressor_wear(comp_power, disc_pressure, fan_rpm, supply_temp, start, dur):
    """
    Compressor Wear — slow progressive degradation over 500+ samples.
    Power creeps up, pressure slowly falls, temp slowly rises; fan normal.
    """
    ramp = np.linspace(0, 1, dur)
    comp_power[start:start + dur]    += ramp * np.random.uniform(1.5, 2.5)
    disc_pressure[start:start + dur] -= ramp * np.random.uniform(30.0, 60.0)
    supply_temp[start:start + dur]   += ramp * np.random.uniform(2.0, 4.5)
    return comp_power, disc_pressure, fan_rpm, supply_temp


# ─── Per-machine data generation ──────────────────────────────────────────────

def generate_machine_data(machine_id, n_samples):
    t      = np.arange(n_samples) * DT
    labels = np.array(["normal"] * n_samples, dtype=object)

    comp_power, disc_pressure, fan_rpm, supply_temp = normal_signals(n_samples, t)
    anomaly_regions = []

    def find_clear_window(size, existing, n, min_gap=100):
        for _ in range(2000):
            s = np.random.randint(100, n - size - 100)
            ok = all(
                s + size + min_gap < r[0] or s - min_gap > r[1]
                for r in existing
            )
            if ok:
                return s
        return None

    # ── Refrigerant leaks (4-6 events, 80-130 samples ≈ 8-13 s) ─────────────
    for _ in range(np.random.randint(4, 7)):
        dur = np.random.randint(80, 131)
        s   = find_clear_window(dur, anomaly_regions, n_samples)
        if s is None:
            continue
        comp_power, disc_pressure, fan_rpm, supply_temp = inject_refrigerant_leak(
            comp_power, disc_pressure, fan_rpm, supply_temp, s, dur
        )
        labels[s:s + dur] = "refrigerant_leak"
        anomaly_regions.append((s, s + dur))

    # ── Fan failures (3-5 events, 100-150 samples) ───────────────────────────
    for _ in range(np.random.randint(3, 6)):
        dur = np.random.randint(100, 151)
        s   = find_clear_window(dur, anomaly_regions, n_samples)
        if s is None:
            continue
        comp_power, disc_pressure, fan_rpm, supply_temp = inject_fan_failure(
            comp_power, disc_pressure, fan_rpm, supply_temp, s, dur
        )
        labels[s:s + dur] = "fan_failure"
        anomaly_regions.append((s, s + dur))

    # ── Compressor wear (3-4 events, 500-700 samples — long gradual drift) ───
    for _ in range(np.random.randint(3, 5)):
        dur = np.random.randint(500, 701)
        s   = find_clear_window(dur, anomaly_regions, n_samples)
        if s is None:
            continue
        comp_power, disc_pressure, fan_rpm, supply_temp = inject_compressor_wear(
            comp_power, disc_pressure, fan_rpm, supply_temp, s, dur
        )
        labels[s:s + dur] = "compressor_wear"
        anomaly_regions.append((s, s + dur))

    return pd.DataFrame({
        "timestamp":              np.round(t, 2),
        "machine_id":             machine_id,
        "compressor_power_kw":    np.round(comp_power, 4),
        "discharge_pressure_psi": np.round(disc_pressure, 4),
        "fan_rpm":                np.round(fan_rpm, 4),
        "supply_air_temp_c":      np.round(supply_temp, 4),
        "fault_label":            labels,
    })


# ─── Build & save CSV ─────────────────────────────────────────────────────────

frames = [generate_machine_data(mid, SAMPLES_PER_MACHINE) for mid in MACHINES]
df     = pd.concat(frames, ignore_index=True)

assert len(df) >= 10_000
expected_cols = {
    "timestamp", "machine_id", "compressor_power_kw",
    "discharge_pressure_psi", "fan_rpm", "supply_air_temp_c", "fault_label",
}
assert set(df.columns) == expected_cols
assert set(MACHINES).issubset(df["machine_id"].unique())
counts = df["fault_label"].value_counts()
for lbl in ("refrigerant_leak", "fan_failure", "compressor_wear"):
    assert counts.get(lbl, 0) >= 100, f"Only {counts.get(lbl, 0)} samples for {lbl}"

csv_path = Path(__file__).parent / "synthetic_data.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {len(df):,} rows -> {csv_path}")
print("\nFault label distribution:")
print(counts.to_string())
print(f"\nNormal: {counts.get('normal', 0):,}")
print(f"Faults: {len(df) - counts.get('normal', 0):,}")


# ─── Visualisation ────────────────────────────────────────────────────────────

COLORS = {
    "normal":           "#4CAF50",
    "refrigerant_leak": "#FF5722",
    "fan_failure":      "#9C27B0",
    "compressor_wear":  "#FF9800",
}

STREAMS = [
    ("compressor_power_kw",    "Compressor Power (kW)"),
    ("discharge_pressure_psi", "Discharge Pressure (PSI)"),
    ("fan_rpm",                "Fan RPM"),
    ("supply_air_temp_c",      "Supply Air Temp (°C)"),
]

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.3)

for col_idx, machine_id in enumerate(MACHINES):
    sub  = df[df["machine_id"] == machine_id].copy()
    view = sub.iloc[:1000]

    for row_idx, (col_name, ylabel) in enumerate(STREAMS):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.set_facecolor("#161B22")

        for label, color in COLORS.items():
            mask = view["fault_label"] == label
            if mask.any():
                ax.scatter(
                    view.loc[mask, "timestamp"],
                    view.loc[mask, col_name],
                    c=color, s=3, alpha=0.85, zorder=3,
                    label=label if (row_idx == 0) else "",
                )
        ax.plot(view["timestamp"], view[col_name],
                color="#30363D", linewidth=0.5, zorder=1, alpha=0.4)
        ax.set_xlabel("Time (s)", color="#8B949E", fontsize=8)
        ax.set_ylabel(ylabel, color="#8B949E", fontsize=8)
        ax.tick_params(colors="#8B949E", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.set_title(f"{machine_id} — {ylabel}", color="#E6EDF3", fontsize=9, pad=6)
        ax.grid(True, color="#21262D", linewidth=0.4, linestyle="--")

        if row_idx == 0 and col_idx == 0:
            legend_elements = [Patch(facecolor=c, label=l) for l, c in COLORS.items()]
            ax.legend(handles=legend_elements, loc="upper right",
                      fontsize=7, facecolor="#161B22", edgecolor="#30363D",
                      labelcolor="#E6EDF3")

plt.suptitle(
    "Thermo-Twin — Carrier HVAC Synthetic Sensor Data (4 Streams)",
    color="#E6EDF3", fontsize=14, y=0.995, fontweight="bold",
)

plot_path = "synthetic_data_preview.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved -> {plot_path}")
