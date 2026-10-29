from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_lib import (
    OUTPUT,
    PROCESSED,
    add_spatial_treatment,
    event_study,
    fit_fe,
)
from run_main import make_sample, setup_logging, tidy_result
from run_robustness import matched_sample


def exact_land_identity(sample: pd.DataFrame) -> pd.DataFrame:
    data = sample.loc[
        sample["price_re_zpg"].gt(0)
        & sample["area_re_zpg"].gt(0)
        & sample["revenue_re_zpg"].gt(0)
    ].copy()
    data["log_price_positive"] = np.log(data["price_re_zpg"])
    data["log_area_positive"] = np.log(data["area_re_zpg"])
    data["log_revenue_positive"] = np.log(data["revenue_re_zpg"])
    rows = []
    for channel, outcome in [
        ("unit price", "log_price_positive"),
        ("quantity", "log_area_positive"),
        ("total revenue", "log_revenue_positive"),
    ]:
        result = fit_fe(data, outcome)
        rows.append(
            {
                "channel": channel,
                "outcome": outcome,
                **{key: result[key] for key in ["coef", "se", "ci_low", "ci_high", "pvalue", "nobs"]},
            }
        )
    table = pd.DataFrame(rows)
    revenue = table.loc[table["channel"].eq("total revenue"), "coef"].iloc[0]
    quantity = table.loc[table["channel"].eq("quantity"), "coef"].iloc[0]
    price = table.loc[table["channel"].eq("unit price"), "coef"].iloc[0]
    table.attrs["identity_residual"] = float(revenue - quantity - price)
    table.attrs["identity_note"] = (
        "Because price equals revenue divided by area in the raw data, log revenue "
        "minus log area equals log price before winsorization."
    )
    return table


def baseline_balance(sample: pd.DataFrame) -> pd.DataFrame:
    pre = sample.loc[sample["year"].between(2007, 2013)].copy()
    city = pre.groupby("citycode", as_index=False).agg(
        treated=("treated_main", "max"),
        gdp=("gdp_10k", "mean"),
        population=("population_10k", "mean"),
        registered_population=("registered_population_10k", "mean"),
        secondary_share=("secondary_share", "mean"),
        airport_count=("airport_count_50", "first"),
        heliport_count=("heliport_count_50", "first"),
        price=("price_re_zpg", "mean"),
    )
    city["population"] = city["population"].fillna(city["registered_population"])
    for col in [
        "gdp",
        "population",
        "secondary_share",
        "airport_count",
        "heliport_count",
        "price",
    ]:
        if col in ["gdp", "population", "price"]:
            city[f"log_{col}"] = np.log1p(city[col])
        else:
            city[col] = city[col]
    variables = [
        "log_gdp",
        "log_population",
        "secondary_share",
        "airport_count",
        "heliport_count",
        "log_price",
    ]
    rows = []
    for variable in variables:
        treated = city.loc[city["treated"].eq(1), variable].dropna()
        control = city.loc[city["treated"].eq(0), variable].dropna()
        pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        rows.append(
            {
                "variable": variable,
                "treated_mean": float(treated.mean()),
                "control_mean": float(control.mean()),
                "standardized_difference": float(
                    (treated.mean() - control.mean()) / pooled_sd
                )
                if pooled_sd > 0
                else np.nan,
                "treated_n": int(treated.size),
                "control_n": int(control.size),
            }
        )
    return pd.DataFrame(rows)


def exposure_interactions(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    exposures = [
        ("airport_count_25", "Airports within 25 km"),
        ("airport_count_50", "Airports within 50 km"),
        ("large_medium_airport_50", "Large or medium airports within 50 km"),
        ("heliport_count_50", "Heliports within 50 km"),
        ("any_airport_50", "Any airport within 50 km"),
        ("any_heliport_50", "Any heliport within 50 km"),
    ]
    for column, label in exposures:
        work = sample.copy()
        raw = work[column].fillna(0)
        exposure = raw if column.startswith("any_") else np.log1p(raw)
        work["_exposure_c"] = exposure - exposure.mean()
        interaction = f"post_x_{column}"
        work[interaction] = work["post"] * work["_exposure_c"]
        try:
            result = fit_fe(
                work,
                "ln_price",
                extra_columns=[interaction],
                focal_terms=["post", interaction],
            )
            row = {
                "exposure": label,
                "column": column,
                "nobs": result["nobs"],
            }
            row.update(result["focal_coefficients"][interaction])
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def spillover_curve(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in [75, 100, 150, 200, 300, 400, 500]:
        spatial = add_spatial_treatment(
            sample, radius_km=radius, minimum_neighbors=5
        )
        if spatial["citycode"].nunique() < 100:
            continue
        result = fit_fe(
            spatial,
            "ln_price",
            extra_columns=["spatial_post"],
            focal_terms=["post", "spatial_post"],
        )
        row = {
            "radius_km": radius,
            "nobs": result["nobs"],
            "cities": result["cities"],
        }
        row.update(result["focal_coefficients"]["spatial_post"])
        rows.append(row)
    return pd.DataFrame(rows)


def normalized_paths(sample: pd.DataFrame) -> pd.DataFrame:
    matched = matched_sample(sample)
    controls = set(matched["citycode"]) - set(
        matched.loc[matched["treated_main"].eq(1), "citycode"]
    )
    rows = []
    for label, codes in [
        ("Narrow pilots", set(matched.loc[matched["treated_main"].eq(1), "citycode"])),
        ("Matched controls", controls),
    ]:
        group = matched.loc[matched["citycode"].isin(codes)].copy()
        annual = (
            group.groupby("year", as_index=False)["ln_price"]
            .mean()
            .sort_values("year")
        )
        baseline = annual.loc[annual["year"].eq(2013), "ln_price"].iloc[0]
        annual["index_2013"] = np.exp(annual["ln_price"] - baseline) * 100
        annual["group"] = label
        rows.append(annual)
    return pd.concat(rows, ignore_index=True)


def pre_trend_adjusted_models(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, data in [
        ("Full eligible sample", sample),
        ("Covariate-matched sample", matched_sample(sample)),
    ]:
        adjusted = []
        for _, group in data.dropna(subset=["ln_price"]).groupby("citycode"):
            pre = group.loc[group["year"].between(2007, 2013)]
            if len(pre) < 5:
                continue
            slope, intercept = np.polyfit(pre["year"], pre["ln_price"], 1)
            work = group.copy()
            work["price_detrended"] = work["ln_price"] - (
                intercept + slope * work["year"]
            )
            adjusted.append(work)
        stacked = pd.concat(adjusted, ignore_index=True)
        result = fit_fe(stacked, "price_detrended")
        rows.append(
            {
                "sample": label,
                "coef": result["coef"],
                "se": result["se"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "pvalue": result["pvalue"],
                "nobs": result["nobs"],
                "cities": result["cities"],
                "note": (
                    "City-specific linear pre-trends are estimated on 2007-2013 "
                    "and then treated as a sensitivity, not a causal correction."
                ),
            }
        )
    return pd.DataFrame(rows)


def flow_outcomes(sample: pd.DataFrame) -> pd.DataFrame:
    work = sample.sort_values(["citycode", "year"]).copy()
    rows = []
    for column, label in [
        ("industrial_firms", "Industrial firm stock growth"),
        ("nightlight", "Night-light growth"),
        ("construction_land_km2", "Construction-land growth"),
    ]:
        growth = np.log1p(work[column]) - np.log1p(
            work.groupby("citycode")[column].shift(1)
        )
        outcome = f"d_{column}"
        work[outcome] = growth
        result = fit_fe(work.dropna(subset=[outcome]), outcome)
        event = event_study(work.dropna(subset=[outcome]), outcome)
        rows.append(
            {
                "outcome": label,
                "coef": result["coef"],
                "se": result["se"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "pvalue": result["pvalue"],
                "nobs": result["nobs"],
                "cities": result["cities"],
                "event_pretrend_f": event.attrs["pretrend_f"],
                "event_pretrend_p": event.attrs["pretrend_p"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_logging()
    panel = pd.read_csv(PROCESSED / "panel_city_year.csv", low_memory=False)
    sample = make_sample(panel)

    identity = exact_land_identity(sample)
    identity.to_csv(OUTPUT / "land_identity_decomposition.csv", index=False, encoding="utf-8-sig")
    balance = baseline_balance(sample)
    balance.to_csv(OUTPUT / "baseline_balance.csv", index=False, encoding="utf-8-sig")
    interactions = exposure_interactions(sample)
    interactions.to_csv(OUTPUT / "exposure_interactions.csv", index=False, encoding="utf-8-sig")
    spillovers = spillover_curve(sample)
    spillovers.to_csv(OUTPUT / "spillover_curve.csv", index=False, encoding="utf-8-sig")
    paths = normalized_paths(sample)
    paths.to_csv(OUTPUT / "matched_price_paths.csv", index=False, encoding="utf-8-sig")
    pretrend = pre_trend_adjusted_models(sample)
    pretrend.to_csv(
        OUTPUT / "pretrend_adjusted_models.csv", index=False, encoding="utf-8-sig"
    )
    flows = flow_outcomes(sample)
    flows.to_csv(OUTPUT / "flow_outcomes.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "land_identity_residual": identity.attrs["identity_residual"],
        "land_identity_note": identity.attrs["identity_note"],
        "spillover_radii_estimable": spillovers["radius_km"].tolist(),
        "baseline_standardized_differences": balance.set_index("variable")[
            "standardized_difference"
        ].to_dict(),
        "exposure_interactions": interactions.set_index("exposure")[
            ["coef", "ci_low", "ci_high"]
        ].to_dict(orient="index"),
        "pretrend_adjusted_models": pretrend.set_index("sample")[
            ["coef", "ci_low", "ci_high"]
        ].to_dict(orient="index"),
        "flow_outcomes": flows.set_index("outcome")[
            ["coef", "ci_low", "ci_high", "event_pretrend_p"]
        ].to_dict(orient="index"),
        "interpretation": (
            "The exposure and spillover estimates are associations. The facility "
            "snapshot is current and does not identify the timing of local infrastructure."
        ),
    }
    (OUTPUT / "extended_experiments.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
