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

    # get camera location for each video frame image
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
    pos_car = np.dot(dist[:, np.newaxis], rot_unit.T) + pos_intp
    df_car = pd.DataFrame(pos_car, columns=['x', 'y'])

    # update license plate by finding connect component in a similarity graph
    df = pd.concat([df, df_car], axis=1)
    df = df[~df['plate'].isna()]
    plates = df['plate'].unique()
    plate_count = df.groupby(['plate'])['time'].count()
    # build graph as edge list
    lp_rel = []
    for lp_1 in plates:
        m_score = []
        for lp_2 in plates:
            if lp_1 == lp_2:
                m_score.append(0.0)
            else:
                scr = similar(lp_1, lp_2)
                if scr > 0.5:
                    m_score.append(scr)
                else:
                    m_score.append(-1.0)
        lp_m = plates[np.argmax(m_score)]
        if ((lp_1, lp_m) not in lp_rel) and ((lp_m, lp_1) not in lp_rel):
            lp_rel.append((lp_1, lp_m))
    # find connected components
    con_comps = []
    for lp_con in lp_rel:
        new_comp = set()
        new_con_comps = []
        for comp in con_comps:
            if (lp_con[0] in comp) or (lp_con[1] in comp):
                new_comp |= comp
            else:
                new_con_comps.append(comp)
        new_comp |= set(lp_con)
        new_con_comps.append(new_comp)
        con_comps = new_con_comps
    # license plate with maximum occurence is the plate
    # for the component (a.k.a. vehicle)
    map_lp = {}
    for comp in con_comps:
        comp = list(comp)
        counts = plate_count.loc[comp].values
        lp_comp = comp[np.argmax(counts)]
        for lp in comp:
            map_lp[lp] = lp_comp
    for index, row in df.iterrows():
        p = df.loc[index, 'plate']
        df.loc[index, 'plate'] = map_lp[p]
    print("Raw VS. cleaned: %d VS. %d" % (plates.shape[0],
                                          df['plate'].unique().shape[0]))
    print("Raw")
    print(plates,'\n')
    print("Cleaned")
    print(df['plate'].unique())

    # update vehicle location based on updated plate number detection result
    gp = df.groupby(['plate'])
    result = gp['time'].agg(['first', 'last'])
    result['plate'] = gp['plate'].agg(lambda x: x.value_counts().index[0])
    result['x'] = gp['x'].mean()
    result['y'] = gp['y'].mean()
    ll = result.apply(lambda r: transform(utm_11, WGS, r['x'], r['y']), axis=1)
    result['lat'] = ll.apply(lambda x: x[1])
    result['lon'] = ll.apply(lambda x: x[0])
    result = result.sort_values('first')

    # return final result
    result['img'] = gp['img'].first()
    result = result[['plate', 'first', 'last', 'lat', 'lon', 'img']]
    result.columns = ['plate', 'start_vid_t', 'end_vid_t',
                      'lat', 'lon', 'img_path']
    result = result.sort_values('start_vid_t')
    return result


if __name__ == "__main__":
    if len(sys.argv) == 1:
        camera_angle = 290 #70
        view_dist_ext = 1.5
        char_sim = 0.5
        f_lprs = 'tmp2/all_frames_lps.csv'
        f_gps = 'test3.gps'
        lpr_out_f = 'test3.lpr'
        traj_out_f = 'test3.traj'
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
