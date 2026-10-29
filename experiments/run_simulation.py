from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research_lib import OUTPUT


def not_yet_att(
    y: np.ndarray,
    group: np.ndarray,
    years: np.ndarray,
    never_code: int = 9999,
) -> float:
    unique_years = np.unique(years)
    n_cities = len(y) // len(unique_years)
    y_matrix = np.asarray(y, dtype=float).reshape(n_cities, len(unique_years))
    city_group = np.asarray(group).reshape(n_cities, len(unique_years))[:, 0]
    post_values = []
    weights = []
    for g in sorted(set(city_group) - {never_code}):
        base_rows = np.flatnonzero(city_group == g)
        base_col = int(np.flatnonzero(unique_years == g - 1)[0])
        for year in unique_years[unique_years >= g]:
            control_rows = np.flatnonzero(
                (city_group == never_code) | (city_group > year)
            )
            current_col = int(np.flatnonzero(unique_years == year)[0])
            treated_change = (
                y_matrix[base_rows, current_col].mean()
                - y_matrix[base_rows, base_col].mean()
            )
            control_change = (
                y_matrix[control_rows, current_col].mean()
                - y_matrix[control_rows, base_col].mean()
            )
            post_values.append(float(treated_change - control_change))
            weights.append(float(len(base_rows)))
    return float(np.average(post_values, weights=weights))


def twfe_beta(y: np.ndarray, treated: np.ndarray, city: np.ndarray, year: np.ndarray) -> float:
    n_cities = len(np.unique(city))
    n_years = len(np.unique(year))
    y_matrix = np.asarray(y, dtype=float).reshape(n_cities, n_years)
    d_matrix = np.asarray(treated, dtype=float).reshape(n_cities, n_years)
    y_residual = (
        y_matrix
        - y_matrix.mean(axis=1, keepdims=True)
        - y_matrix.mean(axis=0, keepdims=True)
        + y_matrix.mean()
    )
    d_residual = (
        d_matrix
        - d_matrix.mean(axis=1, keepdims=True)
        - d_matrix.mean(axis=0, keepdims=True)
        + d_matrix.mean()
    )
    return float((y_residual * d_residual).sum() / (d_residual**2).sum())


def run_scenario(
    name: str,
    effects: dict[int, float],
    reps: int = 500,
    differential_trends: bool = False,
    seed: int = 20261029,
) -> dict:
    rng = np.random.default_rng(seed)
    n_cities = 300
    years = np.arange(2007, 2020)
    cohorts = np.repeat([2010, 2014, 2018, 9999], [80, 80, 80, 60])
    city = np.repeat(np.arange(n_cities), len(years))
    year = np.tile(years, n_cities)
    group = np.repeat(cohorts, len(years))
    treated = (year >= group) & (group != 9999)
    alpha = np.repeat(rng.normal(0, 0.75, n_cities), len(years))
    year_fe = np.tile(rng.normal(0, 0.08, len(years)), n_cities)
    city_trend = np.zeros(n_cities)
    if differential_trends:
        city_trend[np.isin(np.arange(n_cities), np.flatnonzero(cohorts == 2018))] = 0.025
    trend = np.repeat(city_trend, len(years)) * (year - 2007)
    true_effect = np.zeros(len(year))
    for g, tau in effects.items():
        true_effect[(group == g) & (year >= g)] = tau
    weight = np.array(
        [
            sum(cohorts == g) * int((years >= g).sum())
            for g in effects
        ],
        dtype=float,
    )
    true_atts = np.array([effects[g] for g in effects], dtype=float)
    true_att = float(np.average(true_atts, weights=weight))
    twfe_estimates = []
    modern_estimates = []
    for _ in range(reps):
        noise = rng.normal(0, 0.45, len(year))
        y = alpha + year_fe + trend + true_effect + noise
        twfe_estimates.append(twfe_beta(y, treated, city, year))
        modern_estimates.append(not_yet_att(y, group, year))
    twfe = np.asarray(twfe_estimates)
    modern = np.asarray(modern_estimates)
    return {
        "scenario": name,
        "true_att": true_att,
        "repetitions": reps,
        "differential_trends": differential_trends,
        "twfe": {
            "mean": float(twfe.mean()),
            "bias": float(twfe.mean() - true_att),
            "rmse": float(np.sqrt(np.mean((twfe - true_att) ** 2))),
        },
        "not_yet_att": {
            "mean": float(modern.mean()),
            "bias": float(modern.mean() - true_att),
            "rmse": float(np.sqrt(np.mean((modern - true_att) ** 2))),
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenarios = [
        run_scenario(
            "Homogeneous effect",
            {2010: 0.12, 2014: 0.12, 2018: 0.12},
            differential_trends=False,
        ),
        run_scenario(
            "Heterogeneous effect",
            {2010: 0.04, 2014: 0.12, 2018: 0.24},
            differential_trends=False,
        ),
        run_scenario(
            "Heterogeneous effect and cohort trend",
            {2010: 0.04, 2014: 0.12, 2018: 0.24},
            differential_trends=True,
        ),
    ]
    result = {
        "purpose": (
            "Method validation only. Synthetic estimates are excluded from the "
            "empirical findings and are used to show when TWFE can fail."
        ),
        "design": {
            "cities": 300,
            "years": [2007, 2019],
            "cohort_sizes": {"2010": 80, "2014": 80, "2018": 80, "never": 60},
            "residual_sd": 0.45,
        },
        "scenarios": scenarios,
    }
    (OUTPUT / "simulation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
