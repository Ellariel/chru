import os, sys
import time
import argparse
from subprocess import Popen as new, CREATE_NEW_CONSOLE

from codes import CODES

models = [
     'llama3.3:70b',
     'deepseek-r1:70b',
     'gpt-oss:120b',
     'qwen3:235b',
]


def run_scenarios(temp, seed, ver, subset, test, base_dir, windows=False):
    """
    Run all.
    """

    threads = []
    params = dict() if not windows else dict(creationflags=CREATE_NEW_CONSOLE)

    for subset in args.subset:
        print('subset:', subset)
        for code in codes[subset].keys():
            for model in models:
                print('\n<-----running scenario----->')
                threads.append(new(f"uv run {os.path.join(base_dir, 'exec.py')} --model {model} --temp {temp} --seed {seed} --ver {ver} --subset {subset} --code {code} --test {test}",
                                            **params))
                time.sleep(5)

    [i.wait() for i in threads]



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--ver', default='v3', type=str)
    parser.add_argument('--test', default=1, type=int)
    parser.add_argument('--temp', default=1.5, type=float)
    parser.add_argument('--seed', default=1313, type=int)
    parser.add_argument('--subset', default=['ch', 'ru'],#None,
                        type=str, nargs='+', help='--subset ru ch')
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir

    print('temp:', args.temp)
    print('seed:', args.seed)
    print('version:', args.ver)
    print('test:', bool(args.test))

    codes = CODES[args.ver]
    if args.subset is None:
        args.subset = codes.keys()

    run_scenarios(args.temp, 
                  args.seed, 
                  args.ver, 
                  args.subset, 
                  args.test,
                  base_dir=base_dir)


