import os, sys
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from codes import CODES
from prompts import base_prompt, parse_output



def run_with_new_instance(client, prompt, text, 
                          model="llama3.3:70b", 
                          temperature=1.5,
                          sleep=0,
                          seed=1313):
    
    random.seed(seed)
    np.random.seed(seed)
    reset_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Forget about previous inquiries, instructions and prompts."}],
    )
    if sleep:
        time.sleep(sleep)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Here's a piece of text, followed by instructions: “{text}”.\n{prompt}"
            },
        ],
        temperature=temperature,
        seed=seed
    )
    return completion.choices[0].message.content


def read_file(f):
    with open(f, 'r', encoding='utf-8') as file:
        return file.read()
    

def save_results(results, file_name):
        results = pd.DataFrame(results)
        if 'confidence' in results:
            results['confidence'] = pd.to_numeric(results['confidence'], 
                                                  errors='coerce')
        cols = [i for i in [
            'model',
            'seed',
            'temperature',
            'version',
            'subset',
            'outlet',
            'hash',
            'datetime',
            'word_count',
            'tested_code',        
            'code_applied',
            'confidence',
            'justification',
            'output',
            'text_raw',
        ] if i in results.columns]
        results = results[cols]
        results.to_csv(file_name + '.csv', sep=';', index=False)
        results.to_excel(file_name + '.xlsx', index=False)


DEF_ENDPOINT = "https://open-webui.lcl.offis.de/api"



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--test', default=1, type=int)
    parser.add_argument('--ver', default='v1', type=str)
    parser.add_argument('--code', default=None, type=str)
    parser.add_argument('--model', default='gpt-oss:120b', type=str) #'gpt-oss:120b', 'llama3.3:70b'
    parser.add_argument('--temp', default=1.5, type=float)
    parser.add_argument('--seed', default=1313, type=int)
    parser.add_argument('--endpoint', default=DEF_ENDPOINT, type=str)
    parser.add_argument('--subset', default=['ch'], # 'ru', 
                        type=str, nargs='+', help='--subset ru ch')
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    codes = CODES[args.ver]

    print('endpoint:', args.endpoint)
    print('model:', args.model)
    print('temp:', args.temp)
    print('seed:', args.seed)
    print('version:', args.ver)


    if args.code is not None:
        print('code:', args.code)

    client = OpenAI(
        base_url = args.endpoint,
        api_key = read_file(os.path.join(base_dir, "apikey.key"))
    )


    
    def run_proccess(subset, code, desc):
            prompt = base_prompt(code,
                                        desc['description'],
                                        desc['key_features'],
                                        desc['keywords'])
            
            if not bool(args.test):
                texts = os.path.join(base_dir, "texts", subset, "df.csv")
            else:
                texts = os.path.join(base_dir, "validation", subset, "df.csv")
            texts = pd.read_csv(texts, sep=';')
            print(f"n = {texts.shape[0]} articles found.")
            n = texts.shape[0]
            
            result_file = os.path.join(results_dir, 
                    f"{args.model}_{args.seed}_{args.temp}_{args.ver}_{subset}_{code}_{args.test}"\
                                    .replace(':', '-'))
            
            if os.path.exists(f"{result_file}.csv"):
                results = pd.read_csv(f"{result_file}.csv", sep=';', na_filter=False)
                n_done = results.shape[0]
            else:
                results = pd.DataFrame()
                n_done = 0

            if n_done < n:
                texts = texts.iloc[n_done:]
                for idx, item in tqdm(texts.iterrows(),  initial=n_done, total=n, 
                                      leave=False, desc=f"{code}/{args.model}"):
                    try:
                        text = item['text_raw']
                        output = run_with_new_instance(client, prompt, text,
                                                    temperature=args.temp,
                                                    model=args.model,
                                                    seed=args.seed)
                        # print(output)
                        r = parse_output(output)

                        r.update({'subset' : subset,
                                'version' : args.ver,
                                'tested_code' : code,
                                'model' : args.model,
                                'temperature' : args.temp,
                                'seed' : args.seed,
                                })
                        r.update(item.to_dict())

                        results = pd.concat([results, pd.DataFrame([r])])

                        if idx % 5 == 0 and len(results):
                            save_results(results, result_file)

                    except Exception as e:
                        print('\n', str(e))
                        break

                if len(results):
                    save_results(results, result_file)
                    print(f"results saved: {result_file}.csv")



    for subset in args.subset: # ru ch
        print('subset:', subset)

        if args.code is not None:
            run_proccess(subset, args.code, codes[subset][args.code])
        else:
            for code, desc in codes[subset].items():
                run_proccess(subset, code, desc)   
