#!/usr/bin/env python3
"""Compute per-pirate win rates at each position (0-3) and overall."""

import json
import sys
from collections import defaultdict

def main():
    # Load data
    with open("pirates.json", "r") as f:
        pirates_data = json.load(f)
    with open("historical_matches.json", "r") as f:
        matches = json.load(f)

    # Build pirate info lookup
    pirate_info = {}
    for p in pirates_data["pirates"]:
        pirate_info[p["name"]] = {
            "strength": p["strength"],
            "weight": p["weight"],
        }

    # Count wins and appearances per (pirate, position)
    # position = index in the pirates array (0-3)
    wins_by_pos = defaultdict(lambda: defaultdict(int))   # pirate -> pos -> wins
    apps_by_pos = defaultdict(lambda: defaultdict(int))   # pirate -> pos -> appearances
    wins_total = defaultdict(int)
    apps_total = defaultdict(int)

    for day in matches:
        for arena in day:
            winner = arena["winner"]
            pirates_list = arena["pirates"]
            for pos, pirate_entry in enumerate(pirates_list):
                name = pirate_entry["name"]
                apps_by_pos[name][pos] += 1
                apps_total[name] += 1
                if name == winner:
                    wins_by_pos[name][pos] += 1
                    wins_total[name] += 1

    # Build rows
    rows = []
    all_pirates = sorted(apps_total.keys(), key=lambda n: wins_total[n] / max(apps_total[n], 1), reverse=True)

    for name in all_pirates:
        total_apps = apps_total[name]
        total_wins = wins_total[name]
        overall_wr = total_wins / total_apps if total_apps > 0 else 0.0

        pos_wr = []
        pos_n = []
        for p in range(4):
            a = apps_by_pos[name][p]
            w = wins_by_pos[name][p]
            wr = w / a if a > 0 else 0.0
            pos_wr.append(wr)
            pos_n.append(a)

        # Ratio of pos3/pos0 win rate
        if pos_wr[0] > 0:
            ratio = pos_wr[3] / pos_wr[0]
        else:
            ratio = float('inf') if pos_wr[3] > 0 else 0.0

        info = pirate_info.get(name, {"strength": "?", "weight": "?"})
        rows.append({
            "name": name,
            "strength": info["strength"],
            "weight": info["weight"],
            "overall_wr": overall_wr,
            "total_wins": total_wins,
            "total_apps": total_apps,
            "pos_wr": pos_wr,
            "pos_n": pos_n,
            "ratio_p3_p0": ratio,
        })

    # Format output
    lines = []
    lines.append("=" * 160)
    lines.append("PIRATE POSITION WIN RATES")
    lines.append("=" * 160)
    lines.append(f"Total match days: {len(matches)}")
    lines.append("")

    # Header
    hdr = (
        f"{'Pirate':<28s} {'Str':>3s} {'Wt':>3s} "
        f"{'Overall':>8s} {'(W/N)':>12s}  "
        f"{'Pos0 WR':>8s} {'(n)':>6s}  "
        f"{'Pos1 WR':>8s} {'(n)':>6s}  "
        f"{'Pos2 WR':>8s} {'(n)':>6s}  "
        f"{'Pos3 WR':>8s} {'(n)':>6s}  "
        f"{'P3/P0':>7s}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in rows:
        ratio_str = f"{r['ratio_p3_p0']:.3f}" if r['ratio_p3_p0'] != float('inf') else "inf"
        line = (
            f"{r['name']:<28s} {r['strength']:>3} {r['weight']:>3} "
            f"{r['overall_wr']:>8.4f} {r['total_wins']:>5d}/{r['total_apps']:<5d}  "
            f"{r['pos_wr'][0]:>8.4f} {r['pos_n'][0]:>5d}   "
            f"{r['pos_wr'][1]:>8.4f} {r['pos_n'][1]:>5d}   "
            f"{r['pos_wr'][2]:>8.4f} {r['pos_n'][2]:>5d}   "
            f"{r['pos_wr'][3]:>8.4f} {r['pos_n'][3]:>5d}   "
            f"{ratio_str:>7s}"
        )
        lines.append(line)

    lines.append("")
    lines.append("=" * 80)
    lines.append("POSITION 3 / POSITION 0  WIN RATE RATIO (sorted by ratio descending)")
    lines.append("=" * 80)
    lines.append(f"{'Pirate':<28s} {'Pos0 WR':>8s} {'Pos3 WR':>8s} {'Ratio P3/P0':>12s}")
    lines.append("-" * 60)

    ratio_rows = sorted(rows, key=lambda r: r['ratio_p3_p0'] if r['ratio_p3_p0'] != float('inf') else 9999, reverse=True)
    for r in ratio_rows:
        ratio_str = f"{r['ratio_p3_p0']:.4f}" if r['ratio_p3_p0'] != float('inf') else "inf"
        lines.append(
            f"{r['name']:<28s} {r['pos_wr'][0]:>8.4f} {r['pos_wr'][3]:>8.4f} {ratio_str:>12s}"
        )

    lines.append("")

    output = "\n".join(lines)
    print(output)

    with open("pirate_position_winrates.txt", "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to pirate_position_winrates.txt")


if __name__ == "__main__":
    main()
