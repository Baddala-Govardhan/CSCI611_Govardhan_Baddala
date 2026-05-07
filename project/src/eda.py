from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io_utils import read_table

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_eda(data_path: str | Path, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    df = read_table(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")

    cancelled = df["cancelled"].astype(int)
    total = len(df)
    n_cancelled = int(cancelled.sum())
    n_completed = total - n_cancelled
    print(f"  Cancelled: {n_cancelled:,} ({n_cancelled/total*100:.1f}%)")
    print(f"  Completed: {n_completed:,} ({n_completed/total*100:.1f}%)")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Completed", "Cancelled"], [n_completed, n_cancelled], color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Balance: Completed vs Cancelled Rides")
    ax.set_ylabel("Number of Trips")
    for bar, val in zip(ax.patches, [n_completed, n_cancelled]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val:,}\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    _save(fig, out_dir / "eda_class_balance.png")

    # 2. Cancellation rate by hour of day
    hourly = df.groupby("pickup_hour")["cancelled"].mean().reset_index()
    hourly.columns = ["hour", "cancellation_rate"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly["hour"], hourly["cancellation_rate"] * 100, color="#4C72B0")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_title("Cancellation Rate by Hour of Day")
    ax.set_xticks(range(0, 24))
    _save(fig, out_dir / "eda_cancel_by_hour.png")

    # 3. Cancellation rate by day of week
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("pickup_dow")["cancelled"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([dow_labels[i] for i in daily["pickup_dow"]], daily["cancelled"] * 100, color="#4C72B0")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_title("Cancellation Rate by Day of Week")
    _save(fig, out_dir / "eda_cancel_by_dow.png")

    # 4. Trip distance distribution
    dist = df["trip_distance"].dropna()
    dist_clip = dist[dist <= dist.quantile(0.99)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dist_clip, bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Count")
    ax.set_title("Trip Distance Distribution (99th percentile)")
    _save(fig, out_dir / "eda_distance_dist.png")

    # 5. Cancellation rate by distance bucket
    df["dist_bucket"] = pd.cut(
        df["trip_distance"].clip(0, df["trip_distance"].quantile(0.99)),
        bins=[0, 1, 2, 5, 10, 20, 999],
        labels=["0-1mi", "1-2mi", "2-5mi", "5-10mi", "10-20mi", "20+mi"],
    )
    dist_cancel = df.groupby("dist_bucket", observed=True)["cancelled"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dist_cancel.index.astype(str), dist_cancel.values * 100, color="#DD8452")
    ax.set_xlabel("Trip Distance Bucket")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_title("Cancellation Rate by Trip Distance")
    _save(fig, out_dir / "eda_cancel_by_distance.png")

    # 6. Trip volume by hour (heatmap: hour × day)
    pivot = df.groupby(["pickup_dow", "pickup_hour"]).size().unstack(fill_value=0)
    pivot.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in pivot.index]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap="Blues", ax=ax, linewidths=0.3)
    ax.set_title("Trip Volume by Day of Week × Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    _save(fig, out_dir / "eda_volume_heatmap.png")

    # 7. Cancellation rate heatmap: hour × day
    pivot_cancel = df.groupby(["pickup_dow", "pickup_hour"])["cancelled"].mean().unstack(fill_value=0)
    pivot_cancel.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in pivot_cancel.index]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot_cancel * 100, cmap="Oranges", ax=ax, linewidths=0.3, fmt=".0f", annot=False)
    ax.set_title("Cancellation Rate (%) by Day of Week × Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    _save(fig, out_dir / "eda_cancel_heatmap.png")

    print(f"\nEDA complete. {len(list(out_dir.glob('eda_*.png')))} plots saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/labeled.parquet")
    parser.add_argument("--out", default="outputs/eda")
    args = parser.parse_args()
    run_eda(args.data, args.out)


if __name__ == "__main__":
    main()
