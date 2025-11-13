import urllib.request
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def main():
    # Download and load data
    url = (
        "https://raw.githubusercontent.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-"
        "Devs/refs/heads/main/data_complete.csv"
    )
    csv_filename = "data_complete.csv"

    print("Downloading data from GitHub...")
    with urllib.request.urlopen(url) as response:
        content = response.read()
    with open(csv_filename, "wb") as f:
        f.write(content)
    print(f"Saved CSV to {csv_filename}")
    print()

    df = pd.read_csv(csv_filename)

    df["total_implementation_time"] = (
        df["initial_implementation_time"] + df["post_review_implementation_time"]
    )

    print("Initial data shape:", df.shape)

    df = df[df["total_implementation_time"] > 0].copy()
    df = df.dropna(
        subset=["dev_id", "issue_id", "ai_treatment", "total_implementation_time"]
    ).copy()
    df = df[df["ai_treatment"].isin([0, 1])].copy()
    df["ai_treatment"] = df["ai_treatment"].astype(int)

    df["log_total_implementation_time"] = np.log(df["total_implementation_time"])

    print("After filtering:")
    print("  Shape:", df.shape)
    print("  Number of tasks:", len(df))
    print("  Number of developers:", df["dev_id"].nunique())
    print("  ai_treatment value counts:")
    print(df["ai_treatment"].value_counts().sort_index())
    print()

    print("Head of key columns:")
    print(
        df[
            [
                "dev_id",
                "issue_id",
                "ai_treatment",
                "total_implementation_time",
                "log_total_implementation_time",
            ]
        ].head()
    )
    print()

    print("Total implementation time summary (seconds):")
    print(df["total_implementation_time"].describe())
    print()
    print("Log total implementation time summary:")
    print(df["log_total_implementation_time"].describe())
    print()

    dev_summary = (
        df.groupby("dev_id")
        .agg(
            n_tasks=("issue_id", "count"),
            n_ai_0=("ai_treatment", lambda x: int((x == 0).sum())),
            n_ai_1=("ai_treatment", lambda x: int((x == 1).sum())),
        )
        .reset_index()
    )
    print("Per-developer task and treatment counts:")
    print(dev_summary.sort_values("dev_id"))
    print()
    print(
        "Number of developers with both AI and no-AI tasks:",
        int(((dev_summary["n_ai_0"] > 0) & (dev_summary["n_ai_1"] > 0)).sum()),
    )
    print()

    # Frequentist mixed model for a sanity check
    print("Fitting statsmodels MixedLM (random intercept and AI slope per dev)...")
    md = smf.mixedlm(
        "log_total_implementation_time ~ ai_treatment",
        data=df,
        groups=df["dev_id"],
        re_formula="~ai_treatment",
    )
    mdf = md.fit(reml=True, method="lbfgs")
    print()
    print("Statsmodels MixedLM summary:")
    print(mdf.summary())
    print()

    fixed_slope = float(mdf.params["ai_treatment"])

    rows_stats = []
    for dev, re_params in mdf.random_effects.items():
        dev_random_slope = float(re_params["ai_treatment"])
        dev_slope = fixed_slope + dev_random_slope
        time_ratio = float(np.exp(dev_slope))
        rows_stats.append(
            {
                "dev_id": dev,
                "slope_log_time": dev_slope,
                "time_ratio_ai_vs_noai": time_ratio,
            }
        )

    dev_effects_statsmodels = pd.DataFrame(rows_stats).sort_values(
        "slope_log_time"
    ).reset_index(drop=True)

    print("Statsmodels per-developer AI effects (log-time slope and time ratio):")
    print(dev_effects_statsmodels)
    print()
    print(
        "Statsmodels: developers with AI speedup (slope < 0):",
        int((dev_effects_statsmodels["slope_log_time"] < 0).sum()),
    )
    print(
        "Statsmodels: developers with AI slowdown (slope > 0):",
        int((dev_effects_statsmodels["slope_log_time"] > 0).sum()),
    )
    print()

    # Bayesian hierarchical model with symmetric, weakly-informative priors
    print("Fitting PyMC hierarchical model (random intercept and AI slope per dev)...")

    dev_codes = df["dev_id"].astype("category")
    dev_idx = dev_codes.cat.codes.values
    dev_categories = dev_codes.cat.categories
    coords = {"dev": dev_categories}

    ai_array = df["ai_treatment"].values
    log_time_array = df["log_total_implementation_time"].values

    with pm.Model(coords=coords) as hierarchical_model:
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=10.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5.0)

        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)

        z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, dims="dev")
        z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, dims="dev")

        alpha_dev = pm.Deterministic(
            "alpha_dev", mu_alpha + z_alpha * sigma_alpha, dims="dev"
        )
        beta_dev = pm.Deterministic(
            "beta_dev", mu_beta + z_beta * sigma_beta, dims="dev"
        )

        sigma_y = pm.HalfNormal("sigma_y", sigma=5.0)

        mu = alpha_dev[dev_idx] + beta_dev[dev_idx] * ai_array

        pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=log_time_array)

        trace = pm.sample(
            draws=2000,
            tune=2000,
            target_accept=0.95,
            chains=4,
            random_seed=123,
            return_inferencedata=True,
        )

    print()
    print("PyMC population-level summaries (mu_beta and sigma_beta):")
    print(az.summary(trace, var_names=["mu_beta", "sigma_beta"]))
    print()

    divergences = trace.sample_stats["diverging"].values
    n_divergent = int(divergences.sum())
    print(f"PyMC: number of divergent transitions: {n_divergent}")

    summary_all = az.summary(
        trace, var_names=["mu_beta", "sigma_beta", "alpha_dev", "beta_dev"]
    )
    max_rhat = float(summary_all["r_hat"].max())
    min_ess_bulk = float(summary_all["ess_bulk"].min())
    print("PyMC diagnostics for key parameters:")
    print(f"  max r_hat across mu_beta, sigma_beta, alpha_dev, beta_dev: {max_rhat:.3f}")
    print(f"  min ess_bulk across these parameters: {min_ess_bulk:.1f}")
    print()

    # Per-dev posterior summaries and aggregate quantities
    beta_da = trace.posterior["beta_dev"].transpose("chain", "draw", "dev")
    beta_stacked = beta_da.stack(sample=("chain", "draw"))
    beta_np = beta_stacked.values

    n_samples, n_dev = beta_np.shape

    rows_pymc = []
    for j, dev in enumerate(dev_categories):
        beta_j = beta_np[:, j]
        mean_beta_j = float(beta_j.mean())
        ratio_j = float(np.exp(mean_beta_j))

        q_low, q_high = np.quantile(beta_j, [0.03, 0.97])
        ratio_low_j = float(np.exp(q_low))
        ratio_high_j = float(np.exp(q_high))

        p_speed_j = float((beta_j < 0.0).mean())
        p_slow_j = float((beta_j > 0.0).mean())

        rows_pymc.append(
            {
                "dev_id": dev,
                "posterior_mean_slope_log_time": mean_beta_j,
                "posterior_mean_time_ratio_ai_vs_noai": ratio_j,
                "central_94pct_ci_lower_time_ratio": ratio_low_j,
                "central_94pct_ci_upper_time_ratio": ratio_high_j,
                "prob_speedup": p_speed_j,
                "prob_slowdown": p_slow_j,
            }
        )

    dev_effects_pymc = pd.DataFrame(rows_pymc).sort_values(
        "posterior_mean_slope_log_time"
    ).reset_index(drop=True)

    expected_speedup = float(dev_effects_pymc["prob_speedup"].sum())
    expected_slowdown = float(dev_effects_pymc["prob_slowdown"].sum())

    num_speedup_per_draw = (beta_np < 0.0).sum(axis=1)
    any_speedup_draw = num_speedup_per_draw > 0
    prob_any_speedup = float(any_speedup_draw.mean())

    print("PyMC per-developer AI effects (posterior means, intervals, probabilities):")
    print(dev_effects_pymc)
    print()
    print(
        f"PyMC: expected number of developers sped up (beta < 0): {expected_speedup:.2f}"
    )
    print(
        f"PyMC: expected number of developers slowed down (beta > 0): {expected_slowdown:.2f}"
    )
    print(
        f"PyMC: posterior probability that at least one developer is sped up (any beta < 0): {prob_any_speedup:.3f}"
    )
    print()

    # Plots saved to disk only

    plt.style.use("default")

    dev_ids_sorted = dev_effects_pymc["dev_id"].astype(str).values
    x = np.arange(len(dev_effects_pymc))

    ratios = dev_effects_pymc["posterior_mean_time_ratio_ai_vs_noai"].values
    lower = dev_effects_pymc["central_94pct_ci_lower_time_ratio"].values
    upper = dev_effects_pymc["central_94pct_ci_upper_time_ratio"].values

    fig1, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(x, ratios, alpha=0.8)
    ax1.errorbar(
        x,
        ratios,
        yerr=[ratios - lower, upper - ratios],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1,
    )
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dev_ids_sorted, rotation=45, ha="right")
    ax1.set_ylabel("AI vs no-AI time ratio")
    ax1.set_title("Per-developer AI time ratio (posterior mean, central 94% interval)")
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig("per_dev_time_ratio_posterior.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    p_speed = dev_effects_pymc["prob_speedup"].values
    ax2.bar(x, p_speed, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dev_ids_sorted, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Posterior P(speedup)")
    ax2.set_title("Per-developer posterior probability of AI speedup (beta < 0)")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("per_dev_prob_speedup.png", dpi=300)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, n_dev + 1.5, 1)
    ax3.hist(
        num_speedup_per_draw,
        bins=bins,
        edgecolor="black",
        align="mid",
        alpha=0.85,
    )
    ax3.set_xlabel("Number of developers sped up in a posterior draw")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Posterior distribution of number of sped-up developers")
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("posterior_num_sped_up_devs.png", dpi=300)

    dev_effects_with_n = dev_effects_pymc.merge(dev_summary, on="dev_id", how="left")

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(
        dev_effects_with_n["n_tasks"].values,
        dev_effects_with_n["posterior_mean_slope_log_time"].values,
        alpha=0.8,
    )
    ax4.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax4.set_xlabel("Number of tasks per developer")
    ax4.set_ylabel("Posterior mean AI slope (log-time)")
    ax4.set_title("Per-developer AI slope vs. number of tasks")
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("per_dev_slope_vs_tasks.png", dpi=300)

    mu_beta_samples = trace.posterior["mu_beta"].values.reshape(-1)
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    ax5.hist(mu_beta_samples, bins=30, edgecolor="black", alpha=0.85)
    ax5.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax5.set_xlabel("mu_beta (population mean AI effect on log-time)")
    ax5.set_ylabel("Posterior density (scaled as counts)")
    ax5.set_title("Posterior distribution of population-level AI effect (mu_beta)")
    ax5.grid(axis="y", alpha=0.3)
    fig5.tight_layout()
    fig5.savefig("population_mu_beta_posterior.png", dpi=300)


if __name__ == "__main__":
    main()
