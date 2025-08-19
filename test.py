import os, sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score



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

    for code in results['tested_code'].drop_duplicates():
        agreement = []
        data = pd.DataFrame()
        for model, d in results[results['tested_code'] == code].groupby(['model']):
            data = d.copy()
            data[model[0]] = data['code_applied'].apply(lambda x: 1 if code.lower() in str(x).lower() else 0)
            agreement.append(data[[model[0]]])
        agreement = pd.concat(agreement, axis=1)
        agreement['agreement'] = agreement[agreement.columns].sum(1)
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
    
        

