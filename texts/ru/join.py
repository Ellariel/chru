import os, sys
from tqdm import tqdm
import pandas as pd
from glob import glob

def cut(s):
    i = -1
    if ' ' in s[:25]:
        i = s[:25].index(' ')
    return s[i+1:]
        

if __name__ == "__main__":
    
    base_dir = os.path.dirname(__file__)
    dir_list = [i for i in glob(base_dir + '/*') if os.path.isdir(i)]

    data = pd.DataFrame()
    for d in tqdm(dir_list):
        # print(d)
        file_list = [i for i in glob(os.path.join(d, '*.csv')) if os.path.isfile(i)]
        for f in file_list:
            print(f)
            data = pd.concat([data, pd.read_csv(f, sep=';')])
    print(f"{len(data)} articles found.")

    data['text_raw'] = data['text_raw'].apply(cut)
    data['hash'] = data['text_raw'].apply(hash)

    data.drop_duplicates(subset=['outlet', 'datetime', 'word_count'],
                         inplace=True)
    print(f"{len(data)} articles without duplicates.")
    data.to_csv(os.path.join(base_dir, 'df.csv'), index=False, sep=';')


