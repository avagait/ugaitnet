import cv2
import numpy as np
import pickle
import argparse
import os
import deepdish as dd
from datasetInfo import getPartitions


def loadVideo(video_path, mode):
	# VideoCapture to extract the frames.
	cap = cv2.VideoCapture(video_path)
	frames = []
	while (cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
			if mode == 'gray':
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				frame = np.expand_dims(frame, axis=0)
				frame = np.expand_dims(frame, axis=3)
			else:
				frame = np.expand_dims(frame, axis=0)

			frames.append(frame)
		else:
			break

	return np.vstack(frames)

# Prepare input
# Input arguments
parser = argparse.ArgumentParser(description='Build RGB or Gray dataset. Note that no data augmentation is applied')

parser.add_argument('--videodir', type=str, required=False,
                    default='PATH/TUM_GAID/video/',
					help='Full path to of directory')

parser.add_argument('--trackdir', type=str, required=False,
                    default='PATH/TUM_GAID_tr/',
                    help='Full path to tracks directory')

parser.add_argument('--outdir', type=str, required=False,
                    default='PATH/TUM_GAID_tf/',
                    help="Full path for output files. Note that one or more folders are created to store de files")

parser.add_argument('--dataset', type=str, required=False,
                    default='tum_gaid',
                    help="tum_gaid or casiab")

parser.add_argument('--immode', type=str, required=False,
                    default='gray',
                    help="rgb or gray")

parser.add_argument('--mode', type=str, required=False,
                    default='train',
                    help="train: training subset; ft: fine-tuning subset; test: test subset")

parser.add_argument('--nframes', type=int, required=False,
                    default=25,
                    help="Number of frames to be stacked")

parser.add_argument('--step', type=int, required=False,
                    default=5,
                    help="Step size in number of frames")

parser.add_argument('--val_perc', type=float, required=False,
                    default=0.0,
                    help="Percentaje of validation samples")

parser.add_argument('--ids_file_path', type=str, required=False,
                    default='PATH/TUM_GAID/labels',
                    help="Folder containing the id list files. Ony for TUM-GAID")

args = parser.parse_args()

videodir = args.videodir
trackdir = args.trackdir
outdir = args.outdir
dataset = args.dataset
mode = args.mode
immode = args.immode
n_frames = args.nframes
step = args.step
perc = args.val_perc
ids_file_path = args.ids_file_path

file_patterns, folders, gaits, set, ids, im_width, im_height, subject_pattern = getPartitions(dataset, mode, immode, n_frames, ids_file_path)

# Initialize some parameters...
np.random.seed(0)
x_scale = 80 / im_width
y_scale = 60 / im_height

# Loop over different partitions. Each one generates a different folder.
for partition in range(len(file_patterns)):
	# Create output folder if necessary
	try:
		os.mkdir(os.path.join(outdir, folders[partition]))
	except:
		pass

	videoId = 1
	labels_ = []
	videoIds_ = []
	gaits_ = []
	frames_ = []
	bbs_ = []
	file_ = []
	cam_ = []
	if immode == 'gray':
		meanSample = np.zeros((60, 60, 25), dtype=np.int64)
	else:
		meanSample = np.zeros((60, 60, 75), dtype=np.int64)
	# Subjects loop
	for id in ids:
		# Walking condition loop.
		for pattern_ix in range(len(file_patterns[partition])):
			# Files path.
			pattern = file_patterns[partition][pattern_ix]
			video_file = os.path.join(videodir, subject_pattern.format(id) + pattern + '.avi')
			track_file = os.path.join(trackdir, subject_pattern.format(id) + pattern + '.pkl')

			# Load files.
			if os.path.exists(video_file) and os.path.exists(track_file):
				im = loadVideo(video_file, immode)
				with open(track_file, 'rb') as f:
					[full_tracks, full_frames] = pickle.load(f)

				# Stack n_frames continuous frames
				if len(full_tracks) > 0:
					full_tracks = full_tracks[0]
					full_frames = full_frames[0]
					sample_id = 1
					for i in range(0, len(full_tracks), step):
						# Get data of the n_frames
						if (i+1 + n_frames) < len(full_tracks):
							positions = full_frames[i:i + n_frames]
							ims = im[positions, :, :, :]
							sub_position_list = full_tracks[i+1:i+1 + n_frames] # We add 1 since the OF starts in 1

							# n_frames loop to compute the centroid of the  detection
							bbs = []
							centroids = []
							for j in range(len(sub_position_list)):
								# Compute position of the BB and its centroid
								x = int(np.round(sub_position_list[j][1] * x_scale))
								y = int(np.round(sub_position_list[j][0] * y_scale))
								xmax = int(np.round(sub_position_list[j][3] * x_scale))
								ymax = int(np.round(sub_position_list[j][2] * y_scale))

								bb_temp = [x, y, xmax, ymax]
								centroids_temp = [(y + ymax) / 2, (x + xmax) / 2]
								bbs.append(bb_temp)
								centroids.append(centroids_temp)

							# Generate final of maps. Note that 30 is the central point of the frame in x-axis.
							dif_bb = 30 - centroids[round(n_frames / 2)][1]
							M = np.float32([[1, 0, dif_bb], [0, 1, 0]])
							resized_ims = np.zeros([60, 60, ims.shape[3] * n_frames], np.int16)
							for k in range(len(ims)):
								resized_im = cv2.resize(ims[k, :, :, :], (80, 60))

								if immode == 'gray':
									resized_ims[:, :, k] = cv2.warpAffine(resized_im, M, (60, 60))
								else:
									resized_ims[:, :, 3 * k:3 * k + 3] = cv2.warpAffine(resized_im, M, (60, 60))

							# Write output file.
							data = dict()
							data['data'] = np.uint8(resized_ims)
							data['label'] = np.uint16(id)
							data['videoId'] = np.uint16(videoId)
							data['gait'] = np.uint8(gaits[partition][pattern_ix])
							data['frames'] = np.uint16(positions)
							data['bbs'] = np.uint8(sub_position_list)
							data['compressFactor'] = np.uint8(0)
							meanSample = meanSample + np.uint8(resized_ims)
							if dataset == 'casiab':
								data['cam'] = int(pattern.split('-')[-1])
							outpath = os.path.join(outdir, folders[partition], subject_pattern.format(id) + pattern + '-{:02d}'.format(sample_id) + '.h5')
							dd.io.save(outpath, data)

							# Append data for the global file
							labels_.append(id)
							videoIds_.append(videoId)
							gaits_.append(gaits[partition][pattern_ix])
							bbs_.append(sub_position_list)
							frames_.append(positions)
							file_.append(subject_pattern.format(id) + pattern + '-{:02d}'.format(sample_id) + '.h5')
							if dataset == 'casiab':
								cam_.append(int(pattern.split('-')[-1]))

							sample_id = sample_id + 1
				else:
					# Write empty output file.
					data = dict()
					data['data'] = np.uint8([])
					data['label'] = np.uint16(id)
					data['videoId'] = np.uint16(videoId)
					data['gait'] = np.uint8(gaits[partition][pattern_ix])
					data['frames'] = np.uint16([])
					data['bbs'] = np.uint8([])
					data['compressFactor'] = np.uint8(0)
					if dataset == 'casiab':
						data['cam'] = int(pattern.split('-')[-1])
					outpath = os.path.join(outdir, folders[partition],
					                       subject_pattern.format(id) + pattern + '-{:02d}'.format(1) + '.h5')
					dd.io.save(outpath, data)

					# Append data for the global file
					labels_.append(id)
					videoIds_.append(videoId)
					gaits_.append(gaits[partition][pattern_ix])
					bbs_.append([])
					frames_.append([])
					file_.append(subject_pattern.format(id) + pattern + '-{:02d}'.format(1) + '.h5')
					if dataset == 'casiab':
						cam_.append(int(pattern.split('-')[-1]))

				videoId = videoId + 1

	# Get train/val/test sets.
	if mode != 'test':
		set_ = np.zeros(len(labels_))
		np.random.seed(0)
		labels = np.uint16(np.asarray(labels_))
		gg = np.uint8(np.asarray(gaits_))
		ulabs = np.unique(labels)
		ugaits = np.unique(gg)
		nval_samples = len(labels_) * perc
		nsamples_per_id_gait = int(nval_samples / (len(ulabs) * len(ugaits)))
		for id in ulabs:
			for gait in ugaits:
				pos_lab = np.where(labels == id)[0]
				pos_gait = np.where(gg == gait)[0]
				common_pos = np.intersect1d(pos_lab, pos_gait)
				np.random.shuffle(common_pos)
				en_pos = len(common_pos) - nsamples_per_id_gait
				train_samples = common_pos[0:en_pos]
				val_samples = common_pos[en_pos:len(common_pos)]
				set_[train_samples] = 1
				set_[val_samples] = 2

		assert np.count_nonzero(set_) == len(set_)
	else:
		set_ = np.zeros(len(labels_)) + 3

	# Write global file
	data = dict()
	data['label'] = np.uint16(np.asarray(labels_))
	data['videoId'] = np.uint16(np.asarray(videoIds_))
	data['gait'] = np.uint8(np.asarray(gaits_))
	data['set'] = np.uint8(set_)
	data['frames'] = frames_
	data['bbs'] = bbs_
	data['compressFactor'] = np.uint8(0)
	data['file'] = file_
	if immode == 'gray':
		data['shape'] = (60, 60, 25)
	else:
		data['shape'] = (60, 60, 75)
	data['mean'] = meanSample / np.float32(len(labels_))
	if dataset == 'casiab':
		data['cam'] = cam_
	outpath = os.path.join(outdir, folders[partition] + '.h5')
	dd.io.save(outpath, data)
