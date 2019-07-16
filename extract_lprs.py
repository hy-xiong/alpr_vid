import sys
from difflib import SequenceMatcher
import numpy as np
import pandas as pd


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def extract_lprs(n_check, sim_exclude, f_lpr):
    df = pd.read_csv(f_lpr, names=['time', 'license'])
    df = df[~df['license'].isna()]
    n_chars = df['license'].apply(lambda x: len(x))
    df = df[(n_chars >=6) & (n_chars <= 7)]
    lpr_occur = df.groupby(['license'])['time'].agg(['count', 'min', 'max'])
    lpr_occur = lpr_occur.sort_values(['min'])
    lpr_occur.columns = ['n_tStep', 'min_t', 'max_t']
    lpr_check = lpr_occur[lpr_occur['n_tStep'] <= n_check].index.values
    result = lpr_occur[lpr_occur['n_tStep'] > n_check].index.values
    remove_lpr = []
    for lpr in lpr_check:
        for lpr_base in result:
            if similar(lpr, lpr_base) > sim_exclude:
                remove_lpr.append(lpr)
    result = np.append(result, np.setdiff1d(lpr_check, remove_lpr))
    return lpr_occur.loc[result]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        n_occur_check = 2
        sim_exclude = 0.5
        f_lprs = 'tmp/all_frames_lps'
        out_f = 'tmp/result'
    else:
        n_occur_check = int(sys.argv[1])
        sim_exclude = float(sys.argv[2])
        f_lprs = sys.argv[3]
        out_f = sys.argv[4]
    res = extract_lprs(n_occur_check, sim_exclude, f_lprs)
    res.to_csv(out_f, index=True)
    print(res)
