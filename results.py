import os, sys
import numpy as np
import pandas as pd
import argparse
from glob import glob

# from sklearn.metrics import f1_score
from scipy.stats import norm
from statsmodels.stats.inter_rater import aggregate_raters
import krippendorff


def krippendorffs_alpha(
    df,
    measurement="nominal",
    return_per_item_agreement=True,
    return_bootstraped_z_score=True,
    n_iter_for_bootstrap=100,
    bootstrap_seed=13,
    method="two-tailed",
):
    """
    Compute Krippendorff' alpha, z-score, and p-value, per-item agreement
    Level of measurement = "nominal", "ordinal", "interval", "ratio"
    Null hypothesis: Agreement is due to chance
    """
    # https://en.wikipedia.org/wiki/Krippendorff's_Alpha
    # https://github.com/pln-fing-udelar/fast-krippendorff

    df = df.astype(
        str
    )  # be careful about NaNs, they are transformed to a category via .astype(str)
    # print(df)
    m = aggregate_raters(df, n_cat=None)[0]  # [:,:-1]
    # print(m)
    alpha = krippendorff.alpha(value_counts=m, level_of_measurement=measurement)

    # z-score and p-value
    z, p = np.nan, np.nan
    if return_bootstraped_z_score:
        np.random.seed(bootstrap_seed)
        null_dist = []
        for _ in range(n_iter_for_bootstrap):
            m = aggregate_raters(df.apply(np.random.permutation, axis=0), n_cat=None)[
                0
            ]  # [:,:-1]
            null_alpha = krippendorff.alpha(
                value_counts=m, level_of_measurement=measurement
            )
            null_dist.append(null_alpha)
        mu, sigma = np.mean(null_dist), np.std(null_dist)
        if sigma > 0:
            z = (alpha - mu) / sigma
            p = norm.sf(np.abs(z))  # one-tailed, agreement > chance
            p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        agreements = {}
        for idx, row in df.astype(str).iterrows():
            row = row.dropna()
            agreements[idx] = (
                np.nan if len(row) <= 1 else row.value_counts().max() / len(row)
            )
        return (
            alpha,
            z,
            p,
            pd.Series(agreements, index=df.index, name="per_item_agreement"),
        )

    return alpha, z, p


def fleiss_kappa(df, return_per_item_agreement=True, method="two-tailed"):
    """
    Compute Fleiss' kappa, z-score, and p-value, per-item agreement
    Null Hypothesis Kappa = 0	Agreement is due to chance
    0.01-0.02	Slight agreement
    0.21-0.40	Fair Agreement
    0.41-0.60	Moderate Agreement
    0.61-0.80	Substantial Agreement
    0.81-1.00	Almost Perfect Agreement
    Negative (Kappa<0)	Agreement less than that expected by chance
    """
    # https://en.wikipedia.org/wiki/Fleiss%27s_kappa
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.aggregate_raters.html

    n_items, n_raters = df.shape
    m = aggregate_raters(
        df.astype(str), n_cat=None
    )[
        0
    ]  # df.astype(str) be careful about NaNs, they are transformed to a category via .astype(str)
    # print(m)
    P_i = (np.sum(m**2, axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )  # Per-item agreement
    P_j = np.sum(m, axis=0) / (n_items * n_raters)  # Category proportions
    P_e = np.sum(P_j**2)  # Expected agreement
    kappa = (
        (np.mean(P_i) - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan
    )  # Fleiss' kappa

    # variance of kappa (approximation)
    term1 = np.sum(P_j**2 * (1 - P_j) ** 2)
    term2 = (1 - P_e) * (np.sum(P_j**3) - P_e * np.sum(P_j**2))
    var_kappa = (term1 - term2) / (n_items * n_raters * (n_raters - 1) * (1 - P_e) ** 2)

    # z-score and p-value
    z, p = np.nan, np.nan
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p = norm.sf(np.abs(z))  # one-sided
        p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        return (
            kappa,
            z,
            p,
            pd.Series(P_i, index=df.index, name="per_item_agreement"),
        )

    return kappa, z, p


WEIGHTS = {
    "code_adapted_sovereignty": {
        "gemma3:27b": 0.546667,
        "gpt-oss:120b": 0.753333,
        "llama3.3:70b": 0.660000,
        "qwen3:235b": 0.745014,
        "deepseek-r1:70b": 0.680000,
        "qwen2.5:72b": 0.720000,
    },
    "code_conditional_loyalty": {
        "gemma3:27b": 0.700717,
        "gpt-oss:120b": 0.753133,
        "llama3.3:70b": 0.800000,
        "qwen3:235b": 0.900000,
        "deepseek-r1:70b": 0.750000,
        "qwen2.5:72b": 0.770000,
    },
    "code_vulnerable_victims": {
        "gemma3:27b": 0.481481,
        "gpt-oss:120b": 0.708287,
        "llama3.3:70b": 0.759259,
        "qwen3:235b": 0.686869,
        "deepseek-r1:70b": 0.666667,
        "qwen2.5:72b": 0.690000,
    },
    "code_complicit_enablers": {
        "gemma3:27b": 0.605128,
        "gpt-oss:120b": 0.777778,
        "llama3.3:70b": 0.888889,
        "qwen3:235b": 0.640693,
        "deepseek-r1:70b": 0.722222,
        "qwen2.5:72b": 0.750000,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, type=str)
    parser.add_argument("--ver", default="v6", type=str)

    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    from codes import CODES

    codes = CODES[args.ver]

    results = pd.DataFrame()
    for subset in codes.keys():
        for code in codes[subset].keys():
            file_list = [
                i for i in glob(results_dir + "/*.csv") if f"{subset}_{code}_0.csv" in i
            ]
            for f in file_list:
                r = pd.read_csv(f, sep=";")
                results = pd.concat([results, r], axis=0)

    if len(results):
        results.to_csv(os.path.join(results_dir, f"full_results.csv"), index=False)
        results.to_excel(os.path.join(results_dir, f"full_results.xlsx"), index=False)

        models = len(results["model"].drop_duplicates())

        for code in results["tested_code"].drop_duplicates():
            agreement = []
            data = pd.DataFrame()
            for model, d in results[results["tested_code"] == code].groupby(["model"]):
                data = d.copy()
                data[model[0]] = data["code_applied"].apply(
                    lambda x: int(code.lower() in str(x)[:30].lower())
                )
                agreement.append(data[[model[0]]])
            agreement = pd.concat(agreement, axis=1)
            model_columns = agreement.columns

            kappa, kappa_z, kappa_p, kappa_serie = fleiss_kappa(agreement)
            alpha, alpha_z, alpha_p, alpha_serie = krippendorffs_alpha(agreement)

            agreement["agreement_sum"] = agreement[agreement.columns].sum(1)
            agreement["code_applied"] = agreement["agreement_sum"].apply(
                lambda x: int(x > round(models / 2))
            )

            for c in model_columns:
                agreement[c] = agreement[c].apply(
                    lambda x: WEIGHTS[code][c] if pd.notna(x) and bool(x) else x
                )

            agreement["w_agreement_sum"] = agreement[model_columns].sum(1)
            agreement["w_code_applied"] = agreement["w_agreement_sum"].apply(
                lambda x: int(x > sum(WEIGHTS[code].values()) / 2)
            )

            agreement["fleiss_kappa_serie"] = kappa_serie
            agreement["fleiss_kappa"] = kappa
            agreement["fleiss_kappa_z"] = kappa_z
            agreement["fleiss_kappa_p_value"] = kappa_p

            agreement["krippendorffs_alpha_serie"] = alpha_serie
            agreement["krippendorffs_alpha"] = alpha
            agreement["krippendorffs_alpha_z"] = alpha_z
            agreement["krippendorffs_alpha_p_value"] = alpha_p

            # agreement['text_raw'] = data['text_raw']
            # agreement['hash'] = data['hash']
            # agreement['subset'] = data['subset']

            texts = os.path.join(base_dir, "texts", data["subset"].iloc[0], "df.csv")
            texts = pd.read_csv(texts, sep=";")

            # agreement = pd.merge(agreement, texts, how='left',
            #                     on='text_raw')
            agreement = pd.concat([agreement, texts], axis=1)

            agreement.reset_index(drop=True)
            agreement.to_csv(
                os.path.join(results_dir, f"full_agreement_{code}.csv"), index=False
            )
            agreement.to_excel(
                os.path.join(results_dir, f"full_agreement_{code}.xlsx"), index=False
            )
