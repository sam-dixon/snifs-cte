import sep
import click
import numpy as np
import pandas as pd
from astropy.io import fits

DARKS = pd.read_csv('hour_dark_info.csv') # all darks with exptime=3600

def get_data(path, amp):
    data, header = fits.getdata(path, header=True)
    if amp == 0:
        data = np.array(data[:, :1024]).byteswap().newbyteorder()
    else:
        data = np.array(data[:, 1024:]).byteswap().newbyteorder()
    return data, header
    data = fits.getdata(path)
    return data

def get_objs(data):
    bg = sep.Background(data)
    bg.subfrom(data)
    objs = sep.extract(data, 1.5, err=bg.globalrms)
    objs = pd.DataFrame.from_records(objs)
    # Remove objects that are flagged
    objs = objs[objs.flag == 0]
    # Ellipticity cut
    objs = objs[1-objs.b/objs.a < 0.2]
    # Avoid objects on the edge of the detector
    objs = objs[(objs.xpeak > 10) & (objs.xpeak < 1014)]
    objs = objs[(objs.ypeak > 10) & (objs.ypeak < 4086)]
    return objs.reset_index()

def get_aves(data, objs):
    averages = []
    for section, subdf in objs.groupby(pd.cut(objs.ypeak, bins=np.linspace(0, 4096, 17))):
        windows = np.array([data[int(obj.ypeak)-10:int(obj.ypeak)+11, int(obj.xpeak)-10:int(obj.xpeak)+11]
                            for _, obj in subdf.iterrows()])
        p_clip_level = np.percentile(windows, 98, axis=0) # clip highest values
        windows[np.where(windows > p_clip_level)] = np.nan
        average = np.nanmean(windows, axis=0)
        averages.append(average)
    return np.array(averages)
    
