import os
import sep
import sys
import pickle
import numpy as np
import pandas as pd
from astropy.io import fits

def get_data(path, amp):
    data = fits.getdata(path)
    start = 1024 * amp
    end = 1024 * (amp + 1)
    data = np.array(data[:, start:end]).byteswap().newbyteorder()
    return data

def get_objs(data):
    bg = sep.Background(data)
    bg_sub_data = data - bg.back()
    objs = sep.extract(bg_sub_data, 1.5, err=bg.globalrms)
    objs = pd.DataFrame.from_records(objs)
    # Remove objects that are flagged
    objs = objs[objs.flag == 0]
    # Ellipticity cut
    objs = objs[1-objs.b/objs.a < 0.2]
    # Avoid objects on the edge of the detector
    objs = objs[(objs.xpeak > 10) & (objs.xpeak < 1014)]
    objs = objs[(objs.ypeak > 10) & (objs.ypeak < 4086)]
    return objs.reset_index()

def get_tails(data, objs):
    tails, peak_vals = [], []
    for _, obj in objs.iterrows():
        x = int(obj.xpeak)
        y = int(obj.ypeak)
        tails.append(data[y-10:y, x][::-1]-data[y+1:y+11, x])
        peak_vals.append(data[y, x])
    return np.array(tails), np.array(peak_vals)

def main(dark_dir):
    year, night = dark_dir.split('/')[-2:]
    all_tails = {'B': {0: {}, 1: {}},
                 'R': {0: {}, 1: {}},
                 'P': {0: {}, 1: {}, 2: {}, 3: {}}}
    for fname in os.listdir(dark_dir):
        if '_25_' not in fname:
            continue
        print(fname)
        frame = int(fname.split('_')[2])
        channel = fname.split('.')[0][-1]
        path = os.path.join(dark_dir, fname)
        amp_list = range(2) if channel in 'BR' else range(4)
        for amp in amp_list:
            data = get_data(path, amp)
            objs = get_objs(data)
            tails, peak_vals = get_tails(data, objs)
            all_tails[channel][amp][frame] = {'tails': tails,
                                              'locs': objs.ypeak,
                                              'peak_vals': peak_vals}
    tail_fname = '{}_{}.pkl'.format(year, night)
    tail_path = os.path.join('/global/cscratch1/sd/sdixon/cte_tails/', tail_fname)
    with open(tail_path, 'wb') as f:
        pickle.dump(all_tails, f)
    
if __name__ == '__main__':
    main(sys.argv[1])
