import os
import time
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
from functools import wraps
from detect import load_yolo, find_vehicle_one_img, load_lp_net, \
	find_lp_one_img, lp_ocr_one_img


def std_suppress(id):
	def std_dec(func):
		@wraps(func)
		def func_wrapper(*args, **kwargs):
			devnull = os.open(os.devnull, os.O_RDWR)
			saved_io = os.dup(id)
			os.dup2(devnull, id)
			res = func(*args, **kwargs)
			os.dup2(saved_io, id)
			os.close(devnull)
			return res
		return func_wrapper
	return std_dec


@std_suppress(2)
def loadnet():
	vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
	vehicle_netcfg = 'data/vehicle-detector/yolo-voc.cfg'
	vehicle_dataset = 'data/vehicle-detector/voc.data'
	ocr_weights = 'data/ocr/ocr-net.weights'
	ocr_netcfg = 'data/ocr/ocr-net.cfg'
	ocr_dataset = 'data/ocr/ocr-net.data'
	vehicle_net, vehicle_meta = load_yolo(
		vehicle_netcfg, vehicle_weights, vehicle_dataset)
	lp_net = load_lp_net("data/lp-detector/wpod-net_update1.h5")
	ocr_net, ocr_meta = load_yolo(ocr_netcfg, ocr_weights, ocr_dataset)
	return vehicle_net, vehicle_meta, lp_net, ocr_net, ocr_meta


@std_suppress(1)
def lpr(p_img, veh_height_ratio, fl_len, sensor_h, veh_h,
		proc_dir, veh_thd, lp_thd, ocr_thd, veh_net,
		veh_meta, lp_net, ocr_net, ocr_meta):
	print "start dectection:"
	st = time.time()
	raw_height = cv2.imread(p_img).shape[0]
	veh_imgs, _ = find_vehicle_one_img(
		p_img, veh_net, veh_meta, proc_dir, veh_thd)
	lp_strs = []
	veh_indices = []
	dists = []
	if veh_imgs:
		img_heights = np.array([cv2.imread(img).shape[0] for img in veh_imgs])
		img_h_ratios = img_heights * 1.0 / raw_height
		indices = np.argwhere(img_h_ratios >= veh_height_ratio).flatten()
		for i_img in indices:
			veh_img = veh_imgs[i_img]
			dist = veh_h / (img_h_ratios[i_img] * sensor_h) * fl_len / 1000.0
			lp_img, _ = find_lp_one_img(veh_img, lp_net, proc_dir, lp_thd)
			if lp_img:
				lp_s = lp_ocr_one_img(lp_img, ocr_net, ocr_meta, ocr_thd)
				lp_strs.append(lp_s)
				veh_indices.append(i_img)
				dists.append(dist)
	print "1 image LPR runtime: %.1fs" % (time.time() - st)
	return lp_strs, dists, veh_indices


def validate_lp(lp_s):
	if len(lp_s) == 6 or len(lp_s) == 7:
		n_digits = sum(c.isdigit() for c in lp_s)
		if len(lp_s) == 7:
			if (n_digits >= 4) and (n_digits != 6):
				return lp_s
		else:
			if (n_digits >= 2) and (n_digits <= 5):
				return lp_s
	return ""


if __name__ == "__main__":
	if len(sys.argv) == 1:
		t = 0
		t_step = 1
		landscape = 1
		height_ratio = 0.4
		focal_length = 4.44
		sensor_height = 4.29
		veh_height = 1600.0
		out_f_lprs = 'all_frames_lps.csv'
		out_dir = "tmp2"
		test_f = "test3.mp4"
	else:
		t = int(sys.argv[1])
		t_step = int(sys.argv[2])
		landscape = int(sys.argv[3])
		height_ratio = float(sys.argv[4])
		focal_length = float(sys.argv[5])
		sensor_height = float(sys.argv[6])
		veh_height = float(sys.argv[7])
		out_f_lprs = sys.argv[8]
		out_dir = sys.argv[9]
		test_f = sys.argv[10]

	root_dir = os.path.normpath(os.path.join(__file__, '../' * 2))
	out_dir = os.path.join(root_dir, out_dir)
	test_f = os.path.join(root_dir, test_f)

	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	st_all = time.time()

	if os.path.exists(test_f):
		# read video
		vidcap = cv2.VideoCapture(test_f)
		fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
		n_fps = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		fm = int(t * fps)
		# set up network
		vehicle_threshold = .5
		lp_threshold = .5
		ocr_threshold = .4
		vehicle_net, vehicle_meta, lp_net, ocr_net, ocr_meta = loadnet()
		# clip frame and detect license plate
		fm_lps = []
		while fm <= n_fps:
			st = time.time()
			vidcap.set(cv2.CAP_PROP_POS_FRAMES, fm)
			ret, img = vidcap.read()
			fm_img = "%s/%d.jpg" % (out_dir, t)
			if img is not None:
				if not landscape:
					img = np.fliplr(np.swapaxes(img, 0, 1))
				cv2.imwrite(fm_img, img)
				all_lps, all_dist, all_veh_index = lpr(
					fm_img, height_ratio, focal_length, sensor_height,
					veh_height, out_dir, vehicle_threshold, lp_threshold,
					ocr_threshold, vehicle_net, vehicle_meta, lp_net,
					ocr_net, ocr_meta)
				print("%ds done, runtime: %.1fs" % (
					t + t_step, time.time() - st))
				for m in xrange(len(all_lps)):
					lp_str = validate_lp(all_lps[m])
					if lp_str:
						lp_img = os.path.join(
							out_dir, "%d_%dcar_lp.png" % (t, all_veh_index[m]))
						lp_img = os.path.relpath(lp_img, root_dir)
						fm_lps.append([t, lp_str, all_dist[m], lp_img])
						print("Plate: %s, dist: %.4f, plate_img: %s" % (
							lp_str, all_dist[m], lp_img))
			t += t_step
			fm = int(t * fps)
		out_csv = "%s/%s" % (out_dir, out_f_lprs)
		df = pd.DataFrame(fm_lps, columns=['time', 'plate', 'dist', 'img'])
		df.to_csv(out_csv, index=False)
		print("total runtime: %.1fs" % (time.time() - st_all))
