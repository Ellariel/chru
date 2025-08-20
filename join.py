import os, sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score
from scipy.stats import norm

from codes import CODES


def fleiss_kappa_with_p(df: pd.DataFrame):
    """
    Compute Fleiss' kappa, per-item agreement, z-score, and p-value.
    Assumes two categories: "for", "no decision".
    """
    # Encode categories
    matrix = df.to_numpy()
    n_items, n_raters = matrix.shape

    # Counts per category per item
    counts = np.zeros((n_items, 2), dtype=int)
    counts[:, 0] = np.sum(matrix == 0, axis=1)
    counts[:, 1] = np.sum(matrix == 1, axis=1)

    # Per-item agreement
    P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Category proportions
    p_j = np.sum(counts, axis=0) / (n_items * n_raters)

    # Expected agreement
    P_e = np.sum(p_j**2)

    # Fleiss' kappa
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan

    # Variance of kappa (approximation)
    term1 = np.sum(p_j**2 * (1 - p_j)**2)
    term2 = (1 - P_e) * (np.sum(p_j**3) - P_e * np.sum(p_j**2))
    var_kappa = (term1 - term2) / (n_items * n_raters * (n_raters - 1) * (1 - P_e)**2)

    # z-score and p-value
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p_value = 2 * (1 - norm.cdf(abs(z)))
    else:
        z, p_value = np.nan, np.nan

    return pd.Series(P_i, index=df.index, name="per_item_agreement"), kappa, z, p_value




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--ver', default='v1', type=str)

    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    codes = CODES[args.ver]

    results = pd.DataFrame()
    for subset in codes.keys(): 
        print(subset)
        for code in codes[subset].keys():
            file_list = [i for i in glob(results_dir + '/*.csv') if f'{subset}_{code}_0.csv' in i]
            for f in file_list:
                r = pd.read_csv(f, sep=';')
                if len(r):
                    results = pd.concat([results, r], axis=0)

    if len(results):

        results.to_csv(os.path.join(results_dir, 'combined_results.csv'), index=False)
        results.to_excel(os.path.join(results_dir, 'combined_results.xlsx'), index=False)

        for code in results['tested_code'].drop_duplicates():
            agreement = []
            data = pd.DataFrame()
            for model, d in results[results['tested_code'] == code].groupby(['model']):
                data = d.copy()
                data[model[0]] = data['code_applied'].apply(lambda x: int(code.lower() in str(x).lower()))
                agreement.append(data[[model[0]]])
            agreement = pd.concat(agreement, axis=1)
            #print(data.columns)
            fleiss_kappa_serie, kappa, z, p_value = fleiss_kappa_with_p(agreement)
            agreement['agreement_sum'] = agreement[agreement.columns].sum(1)
            agreement['fleiss_kappa_serie'] = fleiss_kappa_serie
            agreement['fleiss_kappa'] = kappa
            agreement['fleiss_kappa_z'] = z
            agreement['fleiss_kappa_p_value'] = p_value
            agreement['code'] = code
            agreement['hash'] = data['hash']
            agreement['text_raw'] = data['text_raw']
            agreement.reset_index(drop=True)
            agreement.to_csv(os.path.join(results_dir, f'combined_agreement_{code}.csv'), index=False)
            agreement.to_excel(os.path.join(results_dir, f'combined_agreement_{code}.xlsx'), index=False)

        

    
        

