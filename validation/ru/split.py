import os, sys
import dateparser
from tqdm import tqdm
import pandas as pd
from glob import glob


def read_file(f):
    with open(f, 'r', encoding='utf-8') as file:
        return file.read()



if __name__ == "__main__":
    
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "raw")
    file_list = [i for i in glob(raw_dir + '/*.txt')]

    data = []

    for f in tqdm(file_list):
        doc = read_file(f)
        data.append({'outlet' : os.path.basename(f),
                     'text_raw' : doc.replace('\n', ' ')\
                                .replace('\r', ' ')\
                                .replace('"', ' ')\
                                .replace("'", ' ')\
                                .replace(";", ',')\
                                .replace('  ', ' ')\
                                .replace('  ', ' ')\
                                .replace('  ', ' ')\
                                .replace('  ', ' ')\
                                .strip(),
                            })
    print(f"{len(data)} articles found.")
    if os.path.exists(os.path.join(base_dir, "validation.csv")):
        data = pd.DataFrame(data)
        val = pd.read_csv(os.path.join(base_dir, "validation.csv"))
        data = pd.merge(data, val, left_on='outlet', right_on='file_name', how='left')
    data.to_csv(os.path.join(base_dir, 'df.csv'), index=False, sep=';')
