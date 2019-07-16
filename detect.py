import cv2
import numpy as np
import time
from os.path import splitext, basename
import darknet.python.darknet as dn

from src.label import Label, lwrite
from src.utils import crop_region
from src.keras_utils import load_model, detect_lp
from src.utils import im2single
from src.label import Shape, writeShapes
from src.label import dknet_label_conversion
from src.utils import nms


def load_yolo(dn_net, net_weights, net_meta):
	vehicle_net = dn.load_net(dn_net, net_weights, 0)
	vehicle_meta = dn.load_meta(net_meta)
	return vehicle_net, vehicle_meta


def find_vehicle_one_img(img_path, veh_net, veh_meta, out_dir, veh_thd):
	st = time.time()
	print '\tScanning %s' % img_path
	bname = basename(splitext(img_path)[0])
	R, _ = dn.detect(veh_net, veh_meta, img_path, thresh=veh_thd)
	# R: [name, prob, [x center, y center, width, height]]
	R = [r for r in R if r[0] in ['car', 'bus']]
	out_img = []
	out_label_f = ""
	if len(R):
		Iorig = cv2.imread(img_path)
		WH = np.array(Iorig.shape[1::-1], dtype=float)
		Lcars = []
		for i, r in enumerate(R):
			cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
			tl = np.array([cx - w / 2., cy - h / 2.])
			br = np.array([cx + w / 2., cy + h / 2.])
			label = Label(0, tl, br)
			Icar = crop_region(Iorig, label)
			Lcars.append(label)
			p_img = '%s/%s_%dcar.png' % (out_dir, bname, i)
			cv2.imwrite(p_img, Icar)
			out_img.append(p_img)
		out_label_f = '%s/%s_cars.txt' % (out_dir, bname)
		lwrite(out_label_f, Lcars)
	print '\t\t%d cars found, runtime: %.1fs' % (len(R), time.time() - st)
	return out_img, out_label_f


def load_lp_net(lp_net_path):
	return load_model(lp_net_path)


def find_lp_one_img(img_path, lp_trans_net, out_dir, lp_thd):
	st = time.time()
	print '\t Processing %s' % img_path
	bname = splitext(basename(img_path))[0]
	Ivehicle = cv2.imread(img_path)
	ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
	side = int(ratio * 288.)
	bound_dim = min(side + (side % (2 ** 4)), 608)
	print "\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio),
	Llp, LlpImgs, _ = detect_lp(lp_trans_net, im2single(Ivehicle), bound_dim,
								2 ** 4, (240, 80), lp_thd)
	lp_img = ""
	lp_label_f = ""
	if len(LlpImgs):
		Ilp = LlpImgs[0]
		Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
		Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
		s = Shape(Llp[0].pts)
		lp_img = '%s/%s_lp.png' % (out_dir, bname)
		lp_label_f = '%s/%s_lp.txt' % (out_dir, bname)
		cv2.imwrite(lp_img, Ilp * 255.)
		writeShapes(lp_label_f, [s])
	print "runtime: %.1fs" % (time.time() - st)
	return lp_img, lp_label_f


def lp_ocr_one_img(img_path, ocr_dn_net, ocr_dn_meta, ocr_thd):
	print '\tScanning %s' % img_path,
	st = time.time()
	R, (width, height) = dn.detect(
		ocr_dn_net, ocr_dn_meta, img_path, thresh=ocr_thd, nms=None)
	lp_str = ""
	if len(R):
		L = dknet_label_conversion(R, width, height)
		L = nms(L, .45)
		L.sort(key=lambda x: x.tl()[0])
		lp_str = ''.join([chr(l.cl()) for l in L])
	print 'runtime: %.1f' % (time.time() - st)
	return lp_str
