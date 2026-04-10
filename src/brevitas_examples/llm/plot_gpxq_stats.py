#!/usr/bin/env python
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualise GPxQ error statistics from a YAML file.

Usage::

    python plot_gpxq_stats.py stats.yaml -o plots/ -f png
"""

import argparse
import os

import matplotlib.pyplot as plt
import yaml

METRICS = ["rel_weight_err", "rel_out_err", "fp_rel_out_err"]

METRIC_LABELS = {
    "rel_weight_err": r"Relative weight error $\|Q-W\|_F / \|W\|_F$",
    "rel_out_err": r"Relative output error (matched)",
    "fp_rel_out_err": r"Relative output error (mismatched)",}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_stats(yaml_path):
    """Return ``(layer_names, stats)`` from a YAML file."""
    with open(yaml_path) as f:
        stats = yaml.safe_load(f)
    layer_names = list(stats.keys())
    return layer_names, stats


def _shorten_names(names, prefix=None):
    """Strip the longest common dot-delimited prefix from *names*.

    If *prefix* is given explicitly it is used instead of auto-detection.
    """
    if prefix is not None:
        return [n[len(prefix):].lstrip(".") if n.startswith(prefix) else n for n in names]
    if len(names) <= 1:
        return list(names)
    # Find longest common prefix ending at a '.' boundary
    parts = [n.split(".") for n in names]
    common = []
    for tokens in zip(*parts):
        if len(set(tokens)) == 1:
            common.append(tokens[0])
        else:
            break
    if common:
        drop = len(".".join(common)) + 1  # +1 for the trailing '.'
        return [n[drop:] for n in names]
    return list(names)


def _available_metrics(stats):
    """Return the subset of ``METRICS`` that have at least one non-None
    value across all layers (checking the ``pre_`` variant)."""
    available = []
    for m in METRICS:
        key = f"pre_{m}"
        if any(layer.get(key) is not None for layer in stats.values()):
            available.append(m)
    return available


def _has_different_pre_post(stats, metrics):
    """Return True if any metric differs between pre and post."""
    for layer in stats.values():
        for m in metrics:
            pre = layer.get(f"pre_{m}")
            post = layer.get(f"post_{m}")
            if pre is not None and post is not None and abs(pre - post) > 1e-9:
                return True
    return False


def _get_values(stats, layer_names, prefix, metric):
    """Extract a list of values for a given prefix/metric, with None for
    missing entries."""
    key = f"{prefix}_{metric}"
    return [stats[n].get(key) for n in layer_names]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_errors(layer_names, short_names, stats, metrics, prefix, ax=None):
    """Line plot of errors across layers for a single prefix (pre or post)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots()
    xs = range(len(layer_names))
    for m in metrics:
        vals = _get_values(stats, layer_names, prefix, m)
        ax.plot(xs, vals, marker="o", markersize=3, linewidth=1.2, label=METRIC_LABELS[m])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Error")
    ax.set_title(f"{'Pre' if prefix == 'pre' else 'Post'}-update errors")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if own_fig:
        fig.tight_layout()
    return ax


def _plot_pre_vs_post(layer_names, short_names, stats, metrics, figsize):
    """One subplot per metric comparing pre (dashed) vs post (solid)."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]
    xs = range(len(layer_names))
    for ax, m in zip(axes, metrics):
        pre = _get_values(stats, layer_names, "pre", m)
        post = _get_values(stats, layer_names, "post", m)
        ax.plot(xs, pre, "--", marker="o", markersize=3, linewidth=1, label="pre", alpha=0.8)
        ax.plot(xs, post, "-", marker="s", markersize=3, linewidth=1, label="post", alpha=0.8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
        ax.set_title(METRIC_LABELS[m], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Pre vs post-update errors", fontsize=12)
    fig.tight_layout()
    return fig


def _plot_error_reduction(layer_names, short_names, stats, metrics, figsize):
    """Bar chart of ``(pre - post) / pre`` for each metric."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]
    xs = range(len(layer_names))
    for ax, m in zip(axes, metrics):
        pre = _get_values(stats, layer_names, "pre", m)
        post = _get_values(stats, layer_names, "post", m)
        reduction = []
        for p, q in zip(pre, post):
            if p is not None and q is not None and abs(p) > 1e-12:
                reduction.append((p - q) / p)
            else:
                reduction.append(0.0)
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in reduction]
        ax.bar(list(xs), reduction, color=colors, alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Relative reduction")
        ax.set_title(METRIC_LABELS[m], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Error reduction (pre - post) / pre", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Visualise GPxQ error statistics from a YAML file.")
    p.add_argument("yaml_path", help="Path to the YAML stats file.")
    p.add_argument(
        "-o", "--output", default=".", help="Directory to save plots (default: current dir).")
    p.add_argument(
        "-f", "--format", default="png", choices=["png", "pdf", "svg"], help="Image format.")
    p.add_argument(
        "--prefix",
        default=None,
        help="Strip this prefix from layer names.  Auto-detected if omitted.")
    p.add_argument(
        "--figsize", default="16,6", help="Figure width,height in inches (default: 16,6).")
    return p.parse_args()


def main():
    args = parse_args()
    figsize = tuple(float(x) for x in args.figsize.split(","))
    layer_names, stats = _load_stats(args.yaml_path)
    short_names = _shorten_names(layer_names, args.prefix)
    metrics = _available_metrics(stats)

    if not metrics:
        print("No metrics found in the YAML file.")
        return

    os.makedirs(args.output, exist_ok=True)
    fmt = args.format

    # Plot 1 & 2: Pre / post errors
    for prefix in ("pre", "post"):
        fig, ax = plt.subplots(figsize=figsize)
        _plot_errors(layer_names, short_names, stats, metrics, prefix, ax=ax)
        fig.tight_layout()
        path = os.path.join(args.output, f"{prefix}_errors.{fmt}")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    # Plot 3 & 4: Pre vs post / error reduction (only when they differ)
    if _has_different_pre_post(stats, metrics):
        fig = _plot_pre_vs_post(layer_names, short_names, stats, metrics, figsize)
        path = os.path.join(args.output, f"pre_vs_post.{fmt}")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

        fig = _plot_error_reduction(layer_names, short_names, stats, metrics, figsize)
        path = os.path.join(args.output, f"error_reduction.{fmt}")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")
    else:
        print("Pre and post errors are identical — skipping comparison plots.")


if __name__ == "__main__":
    main()
