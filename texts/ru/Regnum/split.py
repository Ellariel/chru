import os, sys
import dateparser
from tqdm import tqdm
import pandas as pd
from glob import glob


def read_file(f):
    with open(f, 'r', encoding='utf-8') as file:
        return file.read()

DOC_SPLITER = 'Document DLSE'
WORDS_SPLITER = ' words'

if __name__ == "__main__":
    
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, "raw")
    
    outlet = os.path.basename(base_dir)
    file_list = [i for i in glob(raw_dir + '/*.txt')]

    data = []

    for f in tqdm(file_list):
        docs = read_file(f).split(DOC_SPLITER)
        for d in docs:
            words = d.split(WORDS_SPLITER)[0].split('\n')[-1]\
                .replace('\n', '').replace(',', '')
            words = pd.to_numeric(words, errors='coerce')
            if pd.notna(words):
                print(words, 'words,', end='')
                date = d.split(WORDS_SPLITER)[1].split('\n')[1]
                date = dateparser.parse(date)
                print(date)
                data.append({'outlet' : outlet,
                             'datetime' : date,
                             'word_count' : words,
                             'text_raw' : d.replace('\n', ' ')\
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
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(base_dir, f'{outlet}.csv'), index=False, sep=';')
