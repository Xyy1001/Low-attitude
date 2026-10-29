from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_lib import (
    OUTCOMES_INNOVATION,
    OUTPUT,
    PROCESSED,
    add_spatial_treatment,
    event_study,
    fit_fe,
    fit_poisson_fe,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUTPUT.parent.parent / "logs" / "main_analysis.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def tidy_result(result: dict, sample: str, model: str) -> dict:
    return {
        "sample": sample,
        "model": model,
        "outcome": result["outcome"],
        "term": result["term"],
        "coef": result["coef"],
        "se": result["se"],
        "pvalue": result["pvalue"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "nobs": result["nobs"],
        "cities": result["cities"],
        "clusters": result["clusters"],
    }


def make_sample(panel: pd.DataFrame) -> pd.DataFrame:
    sample = panel.loc[
        panel["eligible_main"].eq(1) & panel["year"].between(2007, 2019)
    ].copy()
    sample["ln_gdp_base_trend"] = sample["ln_gdp_base"] * (sample["year"] - 2007)
    sample["ln_pop_base_trend"] = sample["ln_pop_base"] * (sample["year"] - 2007)
    sample["secondary_share_base_trend"] = sample["secondary_share_base"] * (
        sample["year"] - 2007
    )
    exposure = np.log1p(sample["airport_count_50"])
    sample["airport_exposure_c"] = exposure - exposure.mean()
    sample["post_x_exposure"] = sample["post"] * sample["airport_exposure_c"]
    return sample


def plot_event(events: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    labels = {
        "ln_price": "Log land price",
        "ln_area": "Log land area",
        "ln_revenue": "Log land revenue",
    }
    for ax, (key, event) in zip(axes, events.items()):
        ax.axhline(0, color="0.45", lw=0.8)
        ax.axvline(-1, color="0.8", lw=0.8, ls="--")
        ax.errorbar(
            event["event_time"],
            event["coef"],
            yerr=[
                event["coef"] - event["ci_low"],
                event["ci_high"] - event["coef"],
            ],
            fmt="o-",
            color="#315b7d",
            ecolor="#8da6b8",
            capsize=2,
            lw=1.2,
            ms=3,
        )
        ax.set_title(labels[key])
        ax.set_xlabel("Years since 2014 pilot confirmation")
    axes[0].set_ylabel("Coefficient (95% CI)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_event_study.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logging()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PROCESSED / "panel_city_year.csv", low_memory=False)
    sample = make_sample(panel)
    logging.info("Main sample: %s rows, %s cities", len(sample), sample["citycode"].nunique())

    results = []
    event_results: dict[str, pd.DataFrame] = {}
    for outcome in ["ln_price", "ln_area", "ln_revenue"]:
        baseline = fit_fe(sample, outcome)
        results.append(tidy_result(baseline, "2007-2019 core pilots", "TWFE"))
        logging.info(
            "%s TWFE beta=%.4f (SE %.4f), N=%s",
            outcome,
            baseline["coef"],
            baseline["se"],
            baseline["nobs"],
        )

    controls = [
        "ln_gdp_base_trend",
        "ln_pop_base_trend",
        "secondary_share_base_trend",
    ]
    for outcome in ["ln_price", "ln_area", "ln_revenue"]:
        controlled = fit_fe(sample, outcome, extra_columns=controls)
        results.append(tidy_result(controlled, "2007-2019 core pilots", "TWFE + baseline trends"))

    events = {}
    for outcome in ["ln_price", "ln_area", "ln_revenue"]:
        event = event_study(sample, outcome)
        event["outcome"] = outcome
        events[outcome] = event
        event.to_csv(OUTPUT / f"event_study_{outcome}.csv", index=False, encoding="utf-8-sig")
        logging.info(
            "%s event-study pretrend F=%.3f p=%.4f",
            outcome,
            event.attrs["pretrend_f"],
            event.attrs["pretrend_p"],
        )
    plot_event(events)

    # Innovation and activity outcomes. Patent coverage is incomplete, so these
    # models use their own explicit samples and never inherit the land sample size.
    available_outcomes = [
        ("ln_patent_grants", "2017-2019 limited-coverage association", 2017, 2019),
        ("ln_invention_grants", "2017-2019 limited-coverage association", 2017, 2019),
        ("ln_industrial_firms", "available city-year observations", 2007, 2019),
        ("ln_nightlight", "available city-year observations", 2007, 2019),
        ("ln_construction_land", "available city-year observations", 2007, 2019),
    ]
    for outcome, sample_label, year_start, year_end in available_outcomes:
        outcome_sample = sample.loc[sample["year"].between(year_start, year_end)].copy()
        if outcome_sample[outcome].notna().sum() < 100:
            logging.warning("Skipping %s because fewer than 100 observations exist", outcome)
            continue
        result = fit_fe(outcome_sample, outcome)
        results.append(tidy_result(result, sample_label, "TWFE"))
        logging.info(
            "%s beta=%.4f (SE %.4f), N=%s",
            outcome,
            result["coef"],
            result["se"],
            result["nobs"],
        )

    for count_col, label in [
        ("patent_grants", "ln_patent_grants"),
        ("invention_grants", "ln_invention_grants"),
    ]:
        count_sample = sample.loc[sample["year"].between(2017, 2019)].dropna(
            subset=[count_col]
        ).copy()
        if len(count_sample) >= 100:
            try:
                result = fit_poisson_fe(count_sample, count_col)
                result["outcome"] = label
                if np.isfinite(result["se"]) and np.isfinite(result["pvalue"]):
                    results.append(
                        tidy_result(
                            result,
                            "2017-2019 limited-coverage association",
                            "Poisson FE",
                        )
                    )
                else:
                    logging.warning(
                        "Dropping non-converged Poisson inference for %s", count_col
                    )
            except Exception as exc:
                logging.warning("Poisson model failed for %s: %s", count_col, exc)

    interaction = fit_fe(
        sample,
        "ln_price",
        extra_columns=["post_x_exposure"],
        focal_terms=["post", "post_x_exposure"],
    )
    interaction_row = {
        "sample": "2007-2019 core pilots",
        "model": "TWFE airport exposure interaction",
        "outcome": "ln_price",
        "term": "post_x_exposure",
        "nobs": interaction["nobs"],
        "cities": interaction["cities"],
        "clusters": interaction["clusters"],
    }
    interaction_row.update(interaction["focal_coefficients"]["post_x_exposure"])
    results.append(interaction_row)

    med_gdp = sample["ln_gdp_base"].median()
    sample["high_baseline_gdp"] = (sample["ln_gdp_base"] >= med_gdp).astype(float)
    sample["high_airport_exposure"] = (
        sample["airport_exposure_c"] >= 0
    ).astype(float)
    sample["east"] = sample["region"].eq("东部").astype(float)
    for label, mask in [
        ("East", sample["east"].eq(1)),
        ("Central or West", sample["east"].eq(0)),
        ("High baseline GDP", sample["high_baseline_gdp"].eq(1)),
        ("Low baseline GDP", sample["high_baseline_gdp"].eq(0)),
        ("Any airport within 50 km", sample["any_airport_50"].eq(1)),
        ("No airport within 50 km", sample["any_airport_50"].eq(0)),
    ]:
        subset = sample.loc[mask].copy()
        if subset["citycode"].nunique() < 20:
            continue
        result = fit_fe(subset, "ln_price")
        results.append(tidy_result(result, label, "TWFE split sample"))

    spatial = add_spatial_treatment(sample, radius_km=200, minimum_neighbors=2)
    spatial_result = fit_fe(
        spatial,
        "ln_price",
        extra_columns=["spatial_post"],
        focal_terms=["post", "spatial_post"],
    )
    for term in ["spatial_post"]:
        row = {
            "sample": "2007-2019 core pilots, 200 km inverse-distance spillover",
            "model": "TWFE spatial association",
            "outcome": "ln_price",
            "term": term,
            "nobs": spatial_result["nobs"],
            "cities": spatial_result["cities"],
            "clusters": spatial_result["clusters"],
        }
        row.update(spatial_result["focal_coefficients"][term])
        results.append(row)

    table = pd.DataFrame(results)
    table.to_csv(OUTPUT / "main_results.csv", index=False, encoding="utf-8-sig")

    descriptive_cols = [
        "price_re_zpg",
        "area_re_zpg",
        "revenue_re_zpg",
        "ln_price",
        "ln_area",
        "ln_revenue",
        "gdp_10k",
        "population_10k",
        "industrial_firms",
        "patent_grants",
        "invention_grants",
        "nightlight",
        "construction_land_km2",
        "airport_count_50",
        "heliport_count_50",
    ]
    descriptive = (
        sample[descriptive_cols]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T
    )
    descriptive["missing_rate"] = sample[descriptive_cols].isna().mean()
    descriptive.to_csv(OUTPUT / "descriptive_statistics.csv", encoding="utf-8-sig")

    metadata = {
        "sample_rows": int(len(sample)),
        "sample_cities": int(sample["citycode"].nunique()),
        "treated_cities": int(
            sample.loc[sample["treated_main"].eq(1), "citycode"].nunique()
        ),
        "years": [int(sample["year"].min()), int(sample["year"].max())],
        "cluster": "citycode",
        "outcome_transformation": "log1p after full-sample 1/99 winsorization",
        "event_reference": -1,
        "event_endpoints": {"leads": "<= -6", "lags": ">= 5"},
        "pretrend_tests": {
            key: {
                "F": value.attrs["pretrend_f"],
                "p": value.attrs["pretrend_p"],
            }
            for key, value in events.items()
        },
        "minimum_detectable_effect_80pct_two_sided_5pct": {
            "ln_price_1_99": float(
                (1.959963984540054 + 0.8416212335729143)
                * table.loc[
                    (table["outcome"].eq("ln_price"))
                    & (table["model"].eq("TWFE")),
                    "se",
                ].iloc[0]
            ),
            "interpretation": (
                "Minimum absolute log-point effect detectable with the observed "
                "standard error under a normal approximation."
            ),
        },
        "high_tech_missing_rate": float(sample["ln_high_tech_firms"].isna().mean()),
        "spatial_interpretation": (
            "Association only. The inverse-distance treatment of neighboring cities "
            "does not identify a causal spillover, and alternative distance thresholds "
            "must be compared."
        ),
    }
    (OUTPUT / "main_results.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Wrote %s result rows", len(results))


if __name__ == "__main__":
    main()
