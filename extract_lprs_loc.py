import os
import sys
from difflib import SequenceMatcher
from pyproj import Proj, transform
import numpy as np
import pandas as pd


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def extract_loc(t_array, gps_data):
    _xy = gps_data[[0, -1]]
    pct = ((t_array * 1.0) / np.max(t_array))[:, np.newaxis]
    pos = pct * (_xy[1] - _xy[0])[np.newaxis, :] + _xy[0]
    return pos


def proc_detection(c_angle, view_extend_dist, f_lpr, f_traj, f_gps):
    # get walking trajectory
    WGS = Proj(init="epsg:4326")
    utm_11 = Proj(init="epsg:26911")
    df = pd.read_csv(f_lpr)
    df_loc = pd.read_csv(f_gps, names=['lat', 'lon'])
    xy = df_loc.apply(lambda r: transform(WGS, utm_11, r['lon'], r['lat']),
                      axis=1)
    df_loc['x'] = xy.apply(lambda x: x[0])
    df_loc['y'] = xy.apply(lambda x: x[1])
    gps = df_loc[['x', 'y']].values
    ts = df['time'].values
    pos_intp = extract_loc(ts, gps)

    # traj.to_csv(f_traj, index=False)
    traj = np.apply_along_axis(lambda x: transform(utm_11, WGS, x[0], x[1]),
                               1, np.unique(pos_intp, axis=0))
    traj[:, [0, 1]] = traj[:, [1, 0]]
    traj = traj[::-1]
    traj = np.append(traj, np.unique(ts)[:, np.newaxis], axis=1)
    traj = pd.DataFrame(traj, columns=['lat', 'lon', 't_vid'])
    with open(f_traj, 'w') as wrt:
        str_l = []
        for row in traj.values:
            str_l.append("%f,%f" % (row[1], row[0]))
        wrt.write("%s" % ';'.join(str_l))
    print(traj)

    # get vehical location for each image
    unit = pos_intp[-1] - pos_intp[0]
    unit = unit[:, np.newaxis]
    unit = unit / np.linalg.norm(unit)
    dist = df['dist'].values
    dist[dist == -1.0] = np.nan
    dist += view_extend_dist
    th = np.radians(c_angle)
    rot_M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot_unit = np.dot(rot_M, unit)
    print("vehicle loc offset (m)")
    print(np.dot(dist[:, np.newaxis], rot_unit.T))
    pos_car = np.dot(dist[:, np.newaxis], rot_unit.T) + pos_intp
    df_car = pd.DataFrame(pos_car, columns=['x', 'y'])

    # get veh lpr
    df = pd.concat([df, df_car], axis=1)
    # xy_shift = (df[['x', 'y']].values - gps[0])
    # df['xy_shift'] = np.linalg.norm(xy_shift, axis=1)

    # update lpr
    df = df[~df['plate'].isna()]
    plates = df['plate'].unique()
    plate_count = df.groupby(['plate'])['time'].count()
    check_plate = plate_count[plate_count <= 1].index.values
    safe_lp = np.setdiff1d(plates, check_plate)
    map_lp = []
    for lp in check_plate:
        m_score = []
        for lp_b in safe_lp:
            m_score.append(similar(lp, lp_b))
        map_lp.append(safe_lp[np.argmax(m_score)])
    for index, row in df.iterrows():
        p = df.loc[index, 'plate']
        if p in check_plate:
            df.loc[index, 'plate'] = map_lp[np.argwhere(check_plate == p)[0][0]]
    # df['shift_diff'] = (df['xy_shift'] - df['xy_shift'].shift()).abs()
    # df['new_car'] = df['shift_diff'] >= same_veh_dist_thd
    # df.iloc[0, 8] = True
    # dup = df[df['new_car']]['plate'].duplicated()
    # dup_index = dup[dup].index
    # df.loc[dup_index, 'new_car'] = False
    # df['car_flag'] = df['new_car'].cumsum()

    gp = df.groupby(['plate'])
    result = gp['time'].agg(['first', 'last'])
    result['plate'] = gp['plate'].agg(lambda x: x.value_counts().index[0])
    result['x'] = gp['x'].mean()
    result['y'] = gp['y'].mean()
    ll = result.apply(lambda r: transform(utm_11, WGS, r['x'], r['y']), axis=1)
    result['lat'] = ll.apply(lambda x: x[1])
    result['lon'] = ll.apply(lambda x: x[0])
    result['img'] = gp['img'].first()
    result = result[['plate', 'first', 'last', 'lat', 'lon', 'img']]
    result.columns = ['plate', 'start_vid_t', 'end_vid_t',
                      'lat', 'lon', 'img_path']
    result = result.sort_values('start_vid_t')
    return result


if __name__ == "__main__":
    if len(sys.argv) == 1:
        camera_angle = 70
        view_dist_ext = 2.0
        char_sim = 0.5
        f_lprs = 'tmp/all_frames_lps.csv'
        f_gps = 'test2.gps'
        lpr_out_f = 'test2.lpr'
        traj_out_f = 'test2.traj'
    else:
        camera_angle = int(sys.argv[1])
        view_dist_ext = int(sys.argv[2])
        char_sim = float(sys.argv[3])
        f_lprs = sys.argv[4]
        f_gps = sys.argv[5]
        lpr_out_f = sys.argv[6]
        traj_out_f = sys.argv[7]
    root_dir = os.path.normpath(os.path.join(__file__, '../' * 2))
    f_lprs = os.path.join(root_dir, f_lprs)
    f_gps = os.path.join(root_dir, f_gps)
    lpr_out_f = os.path.join(root_dir, lpr_out_f)
    traj_out_f = os.path.join(root_dir, traj_out_f)
    res = proc_detection(
        camera_angle, view_dist_ext, f_lprs, traj_out_f, f_gps)
    res.to_csv(lpr_out_f, index=False)
    print(res)
