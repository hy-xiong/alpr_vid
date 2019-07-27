import sys
from pyproj import Proj, transform
import numpy as np
import pandas as pd


def extract_loc(t_array, gps_data):
    _xy = gps_data[[0, -1]]
    pct = ((t_array * 1.0) / np.max(t_array))[:, np.newaxis]
    pos = pct * (_xy[1] - _xy[0])[np.newaxis, :] + _xy[0]
    return pos


def proc_detection(c_angle, same_veh_dist_thd, f_lpr, f_gps):
    # get vehical location for each image
    WGS = Proj(init="epsg:4326")
    utm_11 = Proj(init="epsg:26911")
    df = pd.read_csv(f_lpr, names=['time', 'license', 'dist'])
    df_loc = pd.read_csv(f_gps, names=['lat', 'lon'])
    xy = df_loc.apply(lambda r: transform(WGS, utm_11, r['lon'], r['lat']),
                      axis=1)
    df_loc['x'] = xy.apply(lambda x: x[0])
    df_loc['y'] = xy.apply(lambda x: x[1])
    gps = df_loc[['x', 'y']].values
    pos_intp = extract_loc(df['time'].values, gps)
    unit = pos_intp[-1] - pos_intp[0]
    unit = unit[:, np.newaxis]
    unit = unit / np.linalg.norm(unit)
    dist = df['dist'].values
    dist[dist == -1.0] = np.nan
    th = np.radians(c_angle)
    rot_M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot_unit = np.dot(rot_M, unit)
    pos_car = np.dot(dist[:, np.newaxis], rot_unit.T) + pos_intp
    df_car = pd.DataFrame(pos_car, columns=['x', 'y'])
    # get veh lpr
    df = pd.concat([df, df_car], axis=1)
    xy_shift = (df[['x', 'y']].values - gps[0])
    df['xy_shift'] = np.linalg.norm(xy_shift, axis=1)
    df = df[~df['license'].isna()]
    df['shift_diff'] = (df['xy_shift'] - df['xy_shift'].shift()).abs()
    df['new_car'] = df['shift_diff'] >= same_veh_dist_thd
    df.loc[0, 'new_car'] = True
    dup = df[df['new_car']]['license'].duplicated()
    dup_index = dup[dup].index
    df.loc[dup_index, 'new_car'] = False
    df['car_flag'] = df['new_car'].cumsum()
    gp = df.groupby(['car_flag'])
    result = gp['time'].agg(['first', 'last'])
    result['plate'] = gp['license'].agg(lambda x: x.value_counts().index[0])
    result['x'] = gp['x'].mean()
    result['y'] = gp['y'].mean()
    ll = result.apply(lambda r: transform(utm_11, WGS, r['x'], r['y']), axis=1)
    result['lat'] = ll.apply(lambda x: x[1])
    result['lon'] = ll.apply(lambda x: x[0])
    result = result[['plate', 'first', 'last', 'lat', 'lon']]
    result.columns = ['plate', 'start_vid_t', 'end_vid_t', 'lat', 'lon']
    return result


if __name__ == "__main__":
    if len(sys.argv) == 1:
        camera_angle = 45
        dist_thd = 0.5
        f_lprs = 'tmp/all_frames_lps'
        f_gps = 'gps'
        out_f = 'test2.csv'
    else:
        camera_angle = int(sys.argv[1])
        dist_thd = float(sys.argv[2])
        f_lprs = sys.argv[3]
        f_gps = sys.argv[4]
        out_f = sys.argv[5]
    res = proc_detection(camera_angle, dist_thd, f_lprs, f_gps)
    res.to_csv(out_f, index=False)
    print(res)
