import os, sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score
from scipy.stats import norm
from statsmodels.stats.inter_rater import aggregate_raters


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
    m = aggregate_raters(df.astype(str), n_cat=None)[0] # be careful about NaNs, they are transformed to a category via .astype(str)
    P_i = (np.sum(m**2, axis=1) - n_raters) / (n_raters * (n_raters - 1)) # Per-item agreement
    P_j = np.sum(m, axis=0) / (n_items * n_raters) # Category proportions
    P_e = np.sum(P_j**2) # Expected agreement
    kappa = (np.mean(P_i) - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan # Fleiss' kappa

    # variance of kappa (approximation)
    term1 = np.sum(P_j**2 * (1 - P_j)**2)
    term2 = (1 - P_e) * (np.sum(P_j**3) - P_e * np.sum(P_j**2))
    var_kappa = (term1 - term2) / (n_items * n_raters * (n_raters - 1) * (1 - P_e)**2)

    # z-score and p-value
    z, p = np.nan, np.nan
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p = norm.sf(np.abs(z)) # one-sided
        p = p * 2 if method == "two-tailed" else p

    if return_per_item_agreement:
        return kappa, z, p, pd.Series(P_i, index=df.index, 
                                            name="per_item_agreement"), 

    return kappa, z, p



if __name__ == "__main__":
    
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")
    valid_dir = os.path.join(base_dir, "validation")

    results = pd.DataFrame()
    for subset in ['ru', 'ch',]: 
        print(subset)
        v = pd.read_csv(os.path.join(valid_dir, subset, "validation.csv"), sep=';')
        for code in v.columns[1:]:
            file_list = [i for i in glob(results_dir + '/*.csv') if f'{subset}_{code}_1.csv' in i]
            for f in file_list:
                r = pd.read_csv(f, sep=';')
                r = pd.merge(r, v, left_on='outlet', right_on='file_name', how='left')
                results = pd.concat([results, r], axis=0)

    results.to_csv(os.path.join(results_dir, f'validation_results.csv'), index=False)
    results.to_excel(os.path.join(results_dir, f'validation_results.xlsx'), index=False)

    models = len(results['model'].drop_duplicates())

    for code in results['tested_code'].drop_duplicates():
        agreement = []
        data = pd.DataFrame()
        for model, d in results[results['tested_code'] == code].groupby(['model']):
            data = d.copy()
            data[model[0]] = data['code_applied'].apply(lambda x: int(code.lower() in str(x).lower()))
            agreement.append(data[[model[0]]])
        agreement = pd.concat(agreement, axis=1)
        kappa, z, p_value, fleiss_kappa_serie = fleiss_kappa(agreement)
        agreement['agreement_sum'] = agreement[agreement.columns].sum(1)
        agreement['code_applied'] = agreement['agreement_sum'].apply(lambda x: int(x >= round(models / 2)))
        agreement['f1_score'] = f1_score(data[code], agreement['code_applied'], average='weighted')
        agreement['fleiss_kappa_serie'] = fleiss_kappa_serie
        agreement['fleiss_kappa'] = kappa
        agreement['fleiss_kappa_z'] = z
        agreement['fleiss_kappa_p_value'] = p_value
        agreement[code] = data[code]
        agreement['text_raw'] = data['text_raw']
        agreement.reset_index(drop=True)
        agreement.to_csv(os.path.join(results_dir, f'validation_agreement_{code}.csv'), index=False)
        agreement.to_excel(os.path.join(results_dir, f'validation_agreement_{code}.xlsx'), index=False)

    
    metrics = []
    for (model, code), data in results.groupby(['model', 'tested_code']):
        data = data.copy()
        data['result'] = data['code_applied'].apply(lambda x: 1 if code.lower() in str(x).lower() else 0)
        metrics.append((model, code, f1_score(data[code], data['result'], average='weighted')))
    
    metrics = pd.DataFrame(metrics, columns=['model', 'code', 'f1_score'])
    metrics.to_csv(os.path.join(results_dir, 'validation_metrics.csv'), index=False)
    metrics.to_excel(os.path.join(results_dir, 'validation_metrics.xlsx'), index=False)
    print(metrics)  

