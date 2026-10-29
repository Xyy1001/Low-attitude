from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from research_lib import PROCESSED, ROOT


ANALYSIS = ROOT / "research_output" / "analysis"
FIGURES = ROOT / "paper" / "figures"

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light_gray": "#D1D5DB",
    "ink": "#17202A",
    "paper": "#FBFBF8",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "figure.dpi": 180,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    fig.savefig(ANALYSIS / name, bbox_inches="tight")
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    face: str,
    edge: str,
    dashed: bool = False,
    title_size: float = 9.5,
    body_size: float = 8.2,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.4",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.69,
        title,
        ha="center",
        va="center",
        weight="bold",
        fontsize=title_size,
        color=COLORS["ink"],
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.31,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["gray"],
    dashed: bool = False,
    width: float = 1.2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=width,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0.0",
        )
    )


def research_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(
        2,
        97,
        "Policy designation, infrastructure, and capitalization",
        fontsize=14,
        weight="bold",
        ha="left",
        va="top",
        color=COLORS["ink"],
    )
    ax.text(
        2,
        92.5,
        "The observed event is an institutional signal. The economic mechanisms require data that are mostly absent.",
        fontsize=9.2,
        ha="left",
        va="top",
        color=COLORS["gray"],
    )
    ax.text(
        2,
        85.5,
        "1  Observed policy timing",
        fontsize=10,
        weight="bold",
        color=COLORS["blue"],
    )
    rounded_box(
        ax,
        (3, 72),
        20,
        10,
        "2010 national reform",
        "Airspace categories\nand phased liberalization",
        "#EAF4FB",
        COLORS["blue"],
    )
    rounded_box(
        ax,
        (27, 72),
        24,
        10,
        "November 2014",
        "Nine explicit city partitions\npublicly confirmed",
        "#DCEFF8",
        COLORS["blue"],
    )
    arrow(ax, (23.4, 77), (26.6, 77), COLORS["blue"])
    rounded_box(
        ax,
        (71, 72),
        25,
        10,
        "Empirical event",
        "Post-2014 indicator\nfor narrow pilot cities",
        "#E8F4EF",
        COLORS["green"],
    )
    arrow(ax, (51.5, 77), (70.6, 77), COLORS["gray"], dashed=True)

    ax.text(
        2,
        64,
        "2  Conditions required for real capitalization",
        fontsize=10,
        weight="bold",
        color=COLORS["vermillion"],
    )
    required = [
        (3, "Facility lifecycle", "Planning, approval,\nconstruction and opening"),
        (28, "Communication and surveillance", "Coverage, latency,\nredundancy and reliability"),
        (53, "Routes and operations", "Authorized corridors,\nflight hours and capacity"),
        (78, "Firm-level response", "Entry, exit, production,\nR&D and patents"),
    ]
    for x, title, body in required:
        rounded_box(
            ax,
            (x, 48),
            19,
            11,
            title,
            body,
            "#FFF4E3",
            COLORS["vermillion"],
            dashed=True,
            title_size=8.8,
            body_size=7.8,
        )
    for start_x, end_x in [(24, 27), (48, 52), (73, 78)]:
        arrow(ax, (start_x, 53.5), (end_x, 53.5), COLORS["vermillion"], dashed=True)
    ax.text(
        52,
        62,
        "not observed at a common city-year frequency",
        ha="center",
        va="center",
        color=COLORS["vermillion"],
        fontsize=8.5,
        style="italic",
    )

    ax.text(
        2,
        41,
        "3  Market and real-activity outcomes",
        fontsize=10,
        weight="bold",
        color=COLORS["green"],
    )
    outcomes = [
        (3, "Unit land price", "p = revenue / area"),
        (28, "Land quantity", "Area offered and sold"),
        (53, "Firm presence", "Above-scale firm stock"),
        (78, "Innovation", "Patents and high-tech firms"),
    ]
    for x, title, body in outcomes:
        rounded_box(
            ax,
            (x, 26),
            19,
            10,
            title,
            body,
            "#E8F4EF",
            COLORS["green"],
            title_size=8.8,
            body_size=7.8,
        )
    for start_x, end_x in [(22.3, 27.6), (47.3, 52.6), (72.3, 77.6)]:
        arrow(ax, (start_x, 33.5), (end_x, 33.5), COLORS["green"])

    rounded_box(
        ax,
        (3, 7),
        45,
        13,
        "Competing interpretations",
        "Access productivity   |   anticipatory option value\nland-supply response   |   noise and privacy disamenity",
        "#F2F2F2",
        COLORS["gray"],
        title_size=9.2,
        body_size=8.1,
    )
    rounded_box(
        ax,
        (52, 7),
        44,
        13,
        "Design response",
        "Audit the treatment, test pre-trends, alter tails and controls,\nand randomize treatment labels before claiming capitalization",
        "#EEF2F7",
        COLORS["blue"],
        title_size=9.2,
        body_size=8.1,
    )
    ax.text(
        50,
        2,
        "Dashed boxes mark mechanisms that the public city-year data cannot identify.",
        ha="center",
        fontsize=7.7,
        color=COLORS["gray"],
        style="italic",
    )
    save(fig, "figure_research_framework.png")


def evidence_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    stages = [
        ("01", "Data assembly", "326 cities\n5,212 city-years", COLORS["blue"]),
        ("02", "Treatment audit", "9 narrow 2014 pilots\nregional alternatives", COLORS["sky"]),
        ("03", "Baseline test", "+0.130 log points\n95% CI [-0.073, 0.333]", COLORS["orange"]),
        ("04", "Identification test", "pre-trend p=0.0007\nmatched estimate -0.063", COLORS["vermillion"]),
        ("05", "Stress tests", "tail rule -0.012\nrandomization p=0.286", COLORS["purple"]),
        ("06", "Decision", "no robust policy\ncapitalization claim", COLORS["green"]),
    ]
    x_positions = [2, 18.5, 35, 51.5, 68, 84.5]
    for x, (number, title, body, color) in zip(x_positions, stages):
        rounded_box(
            ax,
        (x, 28),
            13.2,
            42,
            f"{number}  {title}",
            body,
            "white",
            color,
            title_size=9.1,
            body_size=8.0,
        )
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        arrow(ax, (left + 13.6, 49), (right - 0.5, 49), COLORS["gray"])
    ax.text(
        2,
        88,
        "Evidence sequence used in this study",
        fontsize=15,
        weight="bold",
        color=COLORS["ink"],
        ha="left",
    )
    ax.text(
        2,
        81,
        "Each stage can weaken the interpretation carried by the previous one.",
        fontsize=9.4,
        color=COLORS["gray"],
        ha="left",
    )
    save(fig, "figure_evidence_pipeline.png")


def pilot_map() -> None:
    exposure = pd.read_csv(PROCESSED / "city_airport_exposure.csv")
    treatment = pd.read_csv(PROCESSED / "pilot_treatment.csv")
    airports = pd.read_csv(PROCESSED / "china_airports_snapshot.csv")
    panel_codes = {
        int(code)
        for code in pd.read_csv(PROCESSED / "panel_city_year.csv", usecols=["citycode"])[
            "citycode"
        ].unique()
    }
    exposure = exposure.loc[exposure["citycode"].isin(panel_codes)].copy()
    main = treatment.loc[treatment["include_main"].eq(1)].merge(
        exposure, on="citycode", how="left", suffixes=("", "_exposure")
    )
    alternatives = treatment.loc[treatment["include_main"].eq(0)].merge(
        exposure, on="citycode", how="left", suffixes=("", "_exposure")
    )
    fig, ax = plt.subplots(figsize=(10.2, 7.1))
    ax.set_xlim(72, 136)
    ax.set_ylim(15, 56)
    ax.set_aspect(1 / np.cos(np.deg2rad(35)))
    ax.scatter(
        airports["longitude_deg"],
        airports["latitude_deg"],
        s=7,
        facecolors="#9CA3AF",
        edgecolors="none",
        alpha=0.38,
        zorder=1,
        label="Current airports and heliports",
    )
    ax.scatter(
        exposure["geo_lon"],
        exposure["geo_lat"],
        s=7,
        facecolors="none",
        edgecolors="#CBD5E1",
        linewidths=0.35,
        alpha=0.5,
        zorder=0,
    )
    sizes = 28 + 22 * np.log1p(main["airport_count_50"].fillna(0))
    ax.scatter(
        main["geo_lon"],
        main["geo_lat"],
        s=sizes,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Narrow 2014 pilot city",
    )
    ax.scatter(
        alternatives["geo_lon"],
        alternatives["geo_lat"],
        s=58,
        marker="^",
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Alternative regional pilot coding",
    )
    offsets = {
        "长春市": (0.8, 0.9),
        "沈阳市": (0.6, -1.1),
        "唐山市": (1.0, 0.4),
        "青岛市": (1.0, 0.2),
        "西安市": (-2.2, 1.0),
        "杭州市": (0.8, 0.8),
        "宁波市": (0.8, -0.8),
        "重庆市": (-2.0, 0.7),
        "昆明市": (-2.3, 0.4),
        "广州市": (1.0, 0.2),
        "海口市": (-1.0, 0.8),
        "三亚市": (1.0, -0.8),
    }
    english_names = {
        "长春市": "Changchun",
        "沈阳市": "Shenyang",
        "唐山市": "Tangshan",
        "青岛市": "Qingdao",
        "西安市": "Xi'an",
        "杭州市": "Hangzhou",
        "宁波市": "Ningbo",
        "重庆市": "Chongqing",
        "昆明市": "Kunming",
        "广州市": "Guangzhou",
        "海口市": "Haikou",
        "三亚市": "Sanya",
        "三沙市": "Sansha",
        "儋州市": "Danzhou",
    }
    for _, row in pd.concat([main, alternatives]).iterrows():
        if pd.isna(row["geo_lon"]) or pd.isna(row["geo_lat"]):
            continue
        dx, dy = offsets.get(row["city_name"], (0.8, 0.7))
        ax.annotate(
            english_names.get(row["city_name"], row["city_name"]),
            (row["geo_lon"], row["geo_lat"]),
            xytext=(row["geo_lon"] + dx, row["geo_lat"] + dy),
            fontsize=7.6,
            color=COLORS["ink"],
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#64748B"},
            zorder=6,
        )
    ax.text(
        73,
        52.6,
        "Pilot definitions and current aviation endowment",
        fontsize=14,
        weight="bold",
        ha="left",
        color=COLORS["ink"],
        bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "alpha": 0.88, "pad": 2},
    )
    ax.text(
        73,
        51.2,
        "Circle area is log(1 + airports and heliports within 50 km). Facility points are a current snapshot, not historical treatment.",
        fontsize=7.8,
        ha="left",
        color=COLORS["gray"],
        bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "alpha": 0.88, "pad": 2},
    )
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.grid(color="#E5E7EB", linewidth=0.45, alpha=0.8)
    ax.legend(
        loc="lower left",
        frameon=False,
        handletextpad=0.8,
        borderpad=0.3,
    )
    fig.text(
        0.98,
        0.015,
        "WGS84 points; equirectangular display. No political boundary is drawn.",
        fontsize=7,
        ha="right",
        color=COLORS["gray"],
        style="italic",
    )
    save(fig, "figure_pilot_map.png")


def event_study_plot() -> None:
    panels = [
        ("event_study_ln_price.csv", "Land price, unmatched", "A"),
        ("event_study_matched_sample.csv", "Land price, matched", "B"),
        ("event_study_ln_area.csv", "Land area, unmatched", "C"),
        ("event_study_ln_revenue.csv", "Land revenue, unmatched", "D"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex=True)
    for ax, (filename, title, letter) in zip(axes.flat, panels):
        data = pd.read_csv(ANALYSIS / filename)
        ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
        ax.axvline(-1, color=COLORS["light_gray"], linewidth=0.8, linestyle="--")
        ax.errorbar(
            data["event_time"],
            data["coef"],
            yerr=[
                data["coef"] - data["ci_low"],
                data["ci_high"] - data["coef"],
            ],
            fmt="o-",
            color=COLORS["blue"],
            ecolor=COLORS["sky"],
            elinewidth=1.0,
            capsize=2,
            markersize=3.5,
            linewidth=1.2,
        )
        ax.set_title(f"{letter}  {title}", loc="left", weight="bold")
        ax.set_ylabel("Coefficient (95% CI)")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
        if filename == "event_study_matched_sample.csv":
            ax.text(
                0.03,
                0.06,
                "Pre-trend F = 0.53; p = 0.750",
                transform=ax.transAxes,
                fontsize=7.7,
                color=COLORS["green"],
            )
        else:
            pvals = {
                "event_study_ln_price.csv": "p = 0.001",
                "event_study_ln_area.csv": "p = 0.046",
                "event_study_ln_revenue.csv": "p < 0.001",
            }
            ax.text(
                0.03,
                0.06,
                f"Lead test: {pvals[filename]}",
                transform=ax.transAxes,
                fontsize=7.7,
                color=COLORS["vermillion"],
            )
    for ax in axes[1]:
        ax.set_xlabel("Years since 2014 public confirmation")
    fig.suptitle(
        "Event-study paths and the fragility of parallel trends",
        x=0.01,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "figure_event_study.png")


def robustness_plot() -> None:
    main = pd.read_csv(ANALYSIS / "main_results.csv")
    robust = pd.read_csv(ANALYSIS / "robustness_results.csv")
    leaveout = pd.read_csv(ANALYSIS / "leave_one_treated_out.csv")
    pretrend = pd.read_csv(ANALYSIS / "pretrend_adjusted_models.csv")
    base = main.loc[
        main["outcome"].eq("ln_price") & main["model"].eq("TWFE")
    ].iloc[0]
    trend = main.loc[
        main["outcome"].eq("ln_price")
        & main["model"].eq("TWFE + baseline trends")
    ].iloc[0]
    matched = robust.loc[robust["sample"].eq("Covariate-matched sample")].iloc[0]
    winsor = robust.loc[robust["sample"].eq("5/95 winsorization")].iloc[0]
    ihs = robust.loc[robust["sample"].eq("IHS transformation")].iloc[0]
    no_muni = robust.loc[robust["sample"].eq("Exclude four municipalities")].iloc[0]
    clus_prov = robust.loc[robust["sample"].eq("Cluster by province")].iloc[0]
    window = robust.loc[robust["sample"].eq("2007-2016 window")].iloc[0]
    broad = robust.loc[robust["sample"].eq("Alternative treatment: broad_2014")].iloc[0]
    hainan = robust.loc[robust["sample"].eq("Alternative treatment: hainan_only")].iloc[0]
    shenyang = robust.loc[robust["sample"].eq("Alternative treatment: shenyang_alt")].iloc[0]
    adjusted = pretrend.loc[pretrend["sample"].eq("Full eligible sample")].iloc[0]
    name_map = {
        "长春市": "Changchun",
        "广州市": "Guangzhou",
        "唐山市": "Tangshan",
        "西安市": "Xi'an",
        "青岛市": "Qingdao",
        "杭州市": "Hangzhou",
        "宁波市": "Ningbo",
        "昆明市": "Kunming",
        "重庆市": "Chongqing",
    }
    specification_rows = [
        ("Baseline", base, COLORS["blue"]),
        ("Baseline trends", trend, COLORS["blue"]),
        ("Covariate matched", matched, COLORS["green"]),
        ("5/95 winsorized", winsor, COLORS["green"]),
        ("IHS outcome", ihs, COLORS["green"]),
        ("Exclude municipalities", no_muni, COLORS["gray"]),
        ("Cluster by province", clus_prov, COLORS["gray"]),
        ("2007-2016 window", window, COLORS["gray"]),
        ("City-specific pre-trend adjustment", adjusted, COLORS["green"]),
        ("Broad pilot definition", broad, COLORS["orange"]),
        ("Hainan only", hainan, COLORS["orange"]),
        ("Shenyang only", shenyang, COLORS["orange"]),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.3, 5.2), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    ax = axes[0]
    y = np.arange(len(specification_rows))[::-1]
    for yi, (label, row, color) in zip(y, specification_rows):
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [yi, yi],
            color=color,
            linewidth=1.4,
            alpha=0.65,
        )
        ax.scatter(row["coef"], yi, color=color, s=34, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in specification_rows])
    ax.set_xlim(-0.36, 0.42)
    ax.set_xlabel("Land-price coefficient (95% CI)")
    ax.set_title("A  Specification sensitivity", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)

    ax = axes[1]
    leaveout = leaveout.sort_values("coef")
    y2 = np.arange(len(leaveout))[::-1]
    for yi, (_, row) in zip(y2, leaveout.iterrows()):
        ax.plot(
            [row["coef"] - 1.96 * row["se"], row["coef"] + 1.96 * row["se"]],
            [yi, yi],
            color=COLORS["sky"],
            linewidth=1.2,
        )
        ax.scatter(row["coef"], yi, color=COLORS["blue"], s=25, zorder=3)
    ax.axvline(base["coef"], color=COLORS["orange"], linewidth=0.9, linestyle="--")
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y2)
    ax.set_yticklabels([name_map.get(name, name) for name in leaveout["omitted_city"]])
    ax.set_xlim(-0.12, 0.42)
    ax.set_xlabel("Estimate after omitting one pilot")
    ax.set_title("B  Leave-one-pilot-out", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    ax.text(
        0.04,
        0.06,
        "Dashed line: full-sample estimate",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLORS["orange"],
    )
    fig.suptitle(
        "One positive estimate, several plausible alternative answers",
        x=0.01,
        ha="left",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "figure_robustness.png")


def randomization_plot() -> None:
    draws = pd.read_csv(ANALYSIS / "placebo_draws.csv")
    placebo = json.loads((ANALYSIS / "placebo_results.json").read_text(encoding="utf-8"))
    pre = pd.read_csv(ANALYSIS / "preperiod_placebo_years.csv")
    info = placebo["randomization_inference"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.hist(
        draws["coef"],
        bins=38,
        color=COLORS["sky"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.axvline(info["observed_coef"], color=COLORS["vermillion"], linewidth=1.6)
    for q in [info["pseudo_quantiles"]["0.05"], info["pseudo_quantiles"]["0.95"]]:
        ax.axvline(q, color=COLORS["orange"], linewidth=0.9, linestyle="--")
    ax.set_xlabel("Two-way fixed-effects coefficient")
    ax.set_ylabel("Randomized draws")
    ax.set_title("A  1,000 random treatment assignments", loc="left", weight="bold")
    ax.text(
        0.03,
        0.91,
        f"Observed = {info['observed_coef']:.3f}\nTwo-sided p = {info['two_sided_p']:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=COLORS["ink"],
    )
    ax = axes[1]
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.errorbar(
        pre["fake_treatment_year"],
        pre["coef"],
        yerr=[
            pre["coef"] - pre["ci_low"],
            pre["ci_high"] - pre["coef"],
        ],
        fmt="o-",
        color=COLORS["blue"],
        ecolor=COLORS["sky"],
        capsize=2,
        linewidth=1.2,
        markersize=4,
    )
    ax.set_xlabel("Placebo treatment year")
    ax.set_ylabel("Coefficient (95% CI)")
    ax.set_title("B  Fake treatment dates within the pre-period", loc="left", weight="bold")
    ax.set_xticks(pre["fake_treatment_year"])
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
    fig.tight_layout()
    save(fig, "figure_randomization_placebo.png")


def mechanisms_plot() -> None:
    main = pd.read_csv(ANALYSIS / "main_results.csv")
    identity = pd.read_csv(ANALYSIS / "land_identity_decomposition.csv")
    exposure = pd.read_csv(ANALYSIS / "exposure_interactions.csv")
    spillover = pd.read_csv(ANALYSIS / "spillover_curve.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    ax = axes[0, 0]
    activity = main.loc[
        main["outcome"].isin(
            ["ln_industrial_firms", "ln_nightlight", "ln_construction_land"]
        )
    ].copy()
    labels = {
        "ln_industrial_firms": "Firm stock",
        "ln_nightlight": "Night light",
        "ln_construction_land": "Construction land",
    }
    y = np.arange(len(activity))[::-1]
    for yi, (_, row) in zip(y, activity.iterrows()):
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=COLORS["sky"], linewidth=1.4)
        ax.scatter(row["coef"], yi, color=COLORS["blue"], s=32, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[value] for value in activity["outcome"]])
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("A  Real-activity outcomes", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)

    ax = axes[0, 1]
    y = np.arange(len(identity))[::-1]
    for yi, (_, row) in zip(y, identity.iterrows()):
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=COLORS["sky"], linewidth=1.4)
        ax.scatter(row["coef"], yi, color=COLORS["green"], s=32, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(identity["channel"])
    ax.set_xlabel("Coefficient under exact log-price identity")
    ax.set_title("B  Price versus quantity", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    ax.text(
        0.03,
        0.07,
        "log revenue = log price + log area",
        transform=ax.transAxes,
        fontsize=7.6,
        color=COLORS["gray"],
    )

    ax = axes[1, 0]
    exposure = exposure.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(exposure))
    for yi, (_, row) in zip(y, exposure.iterrows()):
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=COLORS["sky"], linewidth=1.4)
        ax.scatter(row["coef"], yi, color=COLORS["orange"], s=28, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(exposure["exposure"])
    ax.set_xlabel("Treatment × centered exposure coefficient")
    ax.set_title("C  Current aviation endowment", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)

    ax = axes[1, 1]
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.errorbar(
        spillover["radius_km"],
        spillover["coef"],
        yerr=[
            spillover["coef"] - spillover["ci_low"],
            spillover["ci_high"] - spillover["coef"],
        ],
        fmt="o-",
        color=COLORS["purple"],
        ecolor="#D9B8CB",
        capsize=2,
        linewidth=1.2,
    )
    ax.set_xlabel("Neighbor radius (km)")
    ax.set_ylabel("Spatial association coefficient")
    ax.set_title("D  Distance-based spatial associations", loc="left", weight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
    fig.suptitle("Mechanism checks constrain a simple capitalization story", x=0.01, ha="left", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "figure_mechanisms.png")


def flow_outcomes_plot() -> None:
    data = pd.read_csv(ANALYSIS / "flow_outcomes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.7))
    ax = axes[0]
    y = np.arange(len(data))[::-1]
    for yi, (_, row) in zip(y, data.iterrows()):
        ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=COLORS["sky"], linewidth=1.5)
        ax.scatter(row["coef"], yi, color=COLORS["blue"], s=34, zorder=3)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["outcome"])
    ax.set_xlabel("Change-in-growth coefficient (95% CI)")
    ax.set_title("A  Net-flow associations", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    ax = axes[1]
    colors = [
        COLORS["vermillion"] if p < 0.05 else COLORS["green"]
        for p in data["event_pretrend_p"]
    ]
    evidence = -np.log10(data["event_pretrend_p"].clip(lower=1e-12))
    ax.barh(y, evidence, color=colors, height=0.5)
    ax.axvline(-np.log10(0.05), color=COLORS["orange"], linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Pre-trend evidence, -log10(p)")
    ax.set_title("B  Identification diagnostic", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    for yi, pvalue, value in zip(y, data["event_pretrend_p"], evidence):
        label = "<1e-7" if pvalue < 1e-7 else f"{pvalue:.3f}"
        ax.text(value, yi, f" p={label}", va="center", fontsize=7.7)
    ax.text(
        0.97,
        0.06,
        "Dashed line: p = 0.05",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color=COLORS["orange"],
    )
    fig.suptitle(
        "Growth-rate estimates do not repair the pre-trend problem",
        x=0.01,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, "figure_flow_outcomes.png")


def simulation_plot() -> None:
    data = json.loads((ANALYSIS / "simulation_results.json").read_text(encoding="utf-8"))
    scenarios = data["scenarios"]
    labels = [
        "Homogeneous",
        "Heterogeneous",
        "Heterogeneous + trend",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    y = np.arange(len(scenarios))[::-1]
    for yi, scenario in zip(y, scenarios):
        ax.plot(
            [scenario["not_yet_att"]["mean"], scenario["twfe"]["mean"]],
            [yi, yi],
            color=COLORS["light_gray"],
            linewidth=3,
            zorder=1,
        )
        ax.scatter(
            scenario["true_att"],
            yi,
            marker="X",
            s=52,
            color=COLORS["ink"],
            zorder=3,
            label="Target ATT" if yi == y[0] else None,
        )
        ax.scatter(
            scenario["twfe"]["mean"],
            yi,
            s=38,
            color=COLORS["blue"],
            zorder=4,
            label="TWFE" if yi == y[0] else None,
        )
        ax.scatter(
            scenario["not_yet_att"]["mean"],
            yi,
            s=38,
            marker="s",
            color=COLORS["green"],
            zorder=4,
            label="Not-yet ATT" if yi == y[0] else None,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated or target effect")
    ax.set_title("A  Average estimates", loc="left", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)

    ax = axes[1]
    width = 0.34
    x = np.arange(len(scenarios))
    twfe_bias = [item["twfe"]["bias"] for item in scenarios]
    modern_bias = [item["not_yet_att"]["bias"] for item in scenarios]
    ax.bar(x - width / 2, twfe_bias, width, color=COLORS["blue"], label="TWFE")
    ax.bar(x + width / 2, modern_bias, width, color=COLORS["green"], label="Not-yet ATT")
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Homogeneous", "Heterogeneous", "Heterogeneous\n+ trend"])
    ax.set_ylabel("Bias relative to target ATT")
    ax.set_title("B  Direction and size of bias", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
    fig.suptitle("Synthetic cohorts show why estimator choice needs a timing argument", x=0.01, ha="left", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "figure_simulation.png")


def external_calibration_plot() -> None:
    data = json.loads((ANALYSIS / "external_drone_access.json").read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["serviceability_groups"])
    labels = np.where(rows["drone_serviceable"], "Serviceable", "Not serviceable")
    colors = [COLORS["green"] if value else COLORS["gray"] for value in rows["drone_serviceable"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    specs = [
        ("residences", "Residences in analysis", "Count"),
        ("median_ground_distance_miles", "Median ground distance", "Miles"),
        ("median_ground_time_minutes", "Median one-way ground time", "Minutes"),
    ]
    for ax, (column, title, unit) in zip(axes, specs):
        ax.barh(labels, rows[column], color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(unit)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)
        for yi, value in enumerate(rows[column]):
            ax.text(
                value,
                yi,
                f" {value:,.1f}" if column != "residences" else f" {int(value):,}",
                va="center",
                fontsize=7.6,
                color=COLORS["ink"],
            )
    fig.suptitle(
        "A real delivery surface creates uneven task-level access",
        x=0.01,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Virginia Eastern Shore scenario; not China policy evidence.",
        fontsize=7.5,
        color=COLORS["gray"],
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "figure_external_calibration.png")


def data_coverage_plot() -> None:
    panel = pd.read_csv(PROCESSED / "panel_city_year.csv", low_memory=False)
    coverage_vars = {
        "Land price": "price_re_zpg",
        "GDP": "gdp_10k",
        "Firm stock": "industrial_firms",
        "Night light": "nightlight",
        "Construction land": "construction_land_km2",
        "Patent grants": "patent_grants",
    }
    balance = pd.read_csv(ANALYSIS / "baseline_balance.csv")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1, 1.35]}
    )
    ax = axes[0]
    balance = balance.iloc[::-1].reset_index(drop=True)
    labels = {
        "log_gdp": "Log GDP",
        "log_population": "Log population",
        "secondary_share": "Secondary share",
        "airport_count": "Airport count",
        "heliport_count": "Heliport count",
        "log_price": "Pre-period price",
    }
    y = np.arange(len(balance))
    colors = np.where(balance["standardized_difference"].abs() > 0.25, COLORS["vermillion"], COLORS["blue"])
    ax.barh(y, balance["standardized_difference"], color=colors, height=0.58)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    for threshold in [-0.25, 0.25]:
        ax.axvline(threshold, color=COLORS["orange"], linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[value] for value in balance["variable"]])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("A  Baseline imbalance before matching", loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.45)
    ax.text(
        0.97,
        0.05,
        "Dashed lines: |SMD| = 0.25",
        transform=ax.transAxes,
        fontsize=7.5,
        color=COLORS["orange"],
        ha="right",
    )

    ax = axes[1]
    palette = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["sky"],
        COLORS["vermillion"],
    ]
    for (label, column), color in zip(coverage_vars.items(), palette):
        share = panel.groupby("year")[column].apply(lambda x: x.notna().mean()) * 100
        ax.plot(share.index, share.values, marker="o", markersize=2.8, linewidth=1.4, color=color, label=label)
    ax.axvline(2014, color=COLORS["ink"], linewidth=0.9, linestyle="--")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Nonmissing share (%)")
    ax.set_xlabel("Year")
    ax.set_title("B  Outcome coverage by year", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
    fig.tight_layout()
    save(fig, "figure_data_balance_coverage.png")


def synthetic_control_plot() -> None:
    path = pd.read_csv(ANALYSIS / "synthetic_control_path.csv")
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    ax = axes[0]
    ax.plot(
        path["year"],
        path["treated"],
        marker="o",
        markersize=3,
        color=COLORS["blue"],
        linewidth=1.4,
        label="Narrow pilots",
    )
    ax.plot(
        path["year"],
        path["synthetic"],
        marker="o",
        markersize=3,
        color=COLORS["orange"],
        linewidth=1.4,
        label="Synthetic comparison",
    )
    ax.axvline(2014, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean log land price")
    ax.set_title("A  Fitted path", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
    ax = axes[1]
    colors = [
        COLORS["green"] if value >= 0 else COLORS["vermillion"]
        for value in path["gap"]
    ]
    ax.bar(path["year"], path["gap"], color=colors, width=0.72)
    ax.axvline(2014, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Observed minus synthetic")
    ax.set_title("B  Annual gap", loc="left", weight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.45)
    fig.suptitle(
        "Synthetic comparison is a descriptive check, not an inferential estimate",
        x=0.01,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "figure_synthetic_control.png")


def write_visualization_specs() -> None:
    specs = {
        "framework": {
            "task": "explain",
            "mark": ["box", "arrow"],
            "encoding": {
                "shape": "observed versus unobserved mechanism",
                "position": "causal sequence",
                "color": "evidence status",
            },
            "source": "conceptual synthesis; no estimated values",
        },
        "map": {
            "task": "map",
            "mark": ["point", "proportional_symbol"],
            "encoding": {
                "x": "longitude",
                "y": "latitude",
                "size": "log(1 + airport and heliport count within 50 km)",
                "shape": "narrow versus regional pilot coding",
            },
            "projection": "WGS84 coordinates, equirectangular composition with latitude-adjusted aspect",
            "source": "city_airport_exposure.csv and china_airports_snapshot.csv",
        },
        "event_study": {
            "task": "trend and uncertainty",
            "mark": ["line", "point", "interval"],
            "facet": "outcome and matching status",
            "encoding": {"x": "event time", "y": "coefficient", "y_interval": "95% CI"},
        },
        "robustness": {
            "task": "compare",
            "mark": ["point", "interval"],
            "facet": "specification sensitivity and leave-one-out",
            "encoding": {"x": "coefficient", "y": "specification", "color": "design family"},
        },
        "randomization": {
            "task": "distribution and reference",
            "mark": ["histogram", "rule"],
            "encoding": {"x": "coefficient", "y": "count", "color": "observed value"},
        },
        "mechanisms": {
            "task": "compare and map attenuation",
            "mark": ["point", "interval", "line"],
            "facet": "activity, decomposition, exposure, distance",
        },
        "simulation": {
            "task": "compare estimators",
            "mark": ["point", "bar"],
            "encoding": {"x": "effect", "y": "scenario", "shape": "estimator"},
        },
        "external": {
            "task": "compare groups",
            "mark": ["bar"],
            "facet": "counts, distance, time",
        },
        "coverage": {
            "task": "trend",
            "mark": ["line"],
            "encoding": {"x": "year", "y": "nonmissing share", "color": "variable"},
        },
        "flow_outcomes": {
            "task": "compare and diagnose",
            "mark": ["point", "interval", "bar"],
            "facet": "effect and pre-trend p-value",
        },
        "synthetic_control": {
            "task": "trend comparison and reference",
            "mark": ["line", "bar"],
            "encoding": {"x": "year", "y": "log price or gap"},
        },
    }
    (ANALYSIS / "visualization_specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    style()
    research_framework()
    evidence_pipeline()
    pilot_map()
    event_study_plot()
    robustness_plot()
    randomization_plot()
    mechanisms_plot()
    flow_outcomes_plot()
    simulation_plot()
    external_calibration_plot()
    data_coverage_plot()
    synthetic_control_plot()
    write_visualization_specs()
    print("Wrote 12 manuscript figures and visualization specifications.")


if __name__ == "__main__":
    main()
