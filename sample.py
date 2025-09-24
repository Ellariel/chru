import os, sys
import random
import numpy as np
import pandas as pd
import argparse
from glob import glob
#from sklearn.metrics import f1_score
#from scipy.stats import norm
#from statsmodels.stats.inter_rater import aggregate_raters
#import krippendorff



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--ver', default='v4', type=str)

    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    from codes import CODES
    codes = CODES[args.ver]
    sample_size = 10
    random_state = 1313
    
    results = {}
    for subset in codes.keys():
        for code in codes[subset].keys():
            f = os.path.join(results_dir, f'full_agreement_{code}.csv')
            if os.path.exists(f):
                results[code] = pd.read_csv(f)
    for code, res in results.items():
        random.seed(random_state)
        np.random.seed(random_state)
        with_code = res[res['code_applied'].astype(bool)].sample(sample_size, random_state=random_state)
        without_code = res[~res['code_applied'].astype(bool)].sample(sample_size, random_state=random_state)
        d = pd.concat([with_code, without_code])
        d.to_csv(os.path.join(results_dir, f'sample_{code}.csv'), index=False)
        d.to_excel(os.path.join(results_dir, f'sample_{code}.xlsx'), index=False)

    
    
    
    
    
    

