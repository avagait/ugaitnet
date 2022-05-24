import sys
import numpy as np
import os

def getPartitions(dataset, mode, data_type, n_frames, ids_file_path='', pattern=''):
	file_patterns = []
	folders = []
	gaits = []
	set = 1
	if dataset == 'tum_gaid':
		im_width = 640
		im_height = 480
		subject_pattern = 'p{:03d}'

		# Build the list of patterns for reading the files. We use a list of lists structure.
		# The first level is used to define a global set of samples and the second level includes all files of that set.
		# For example: training data for tum and tum temporal.
		if mode == 'train':
			# Load id file.
			file = open(os.path.join(ids_file_path, 'tumgaidtrainvalids.lst'), mode='r')
			ids = file.read()
			ids = ids[0:-1].split('\n')
			ids = [int(i) for i in ids]

			# Normal set
			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N150_train_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[0].append('-n01')
			gaits[0].append(1)
			file_patterns[0].append('-n02')
			gaits[0].append(1)
			file_patterns[0].append('-n03')
			gaits[0].append(1)
			file_patterns[0].append('-n04')
			gaits[0].append(1)
			file_patterns[0].append('-n05')
			gaits[0].append(1)
			file_patterns[0].append('-n06')
			gaits[0].append(1)
			file_patterns[0].append('-b01')
			gaits[0].append(2)
			file_patterns[0].append('-b02')
			gaits[0].append(2)
			file_patterns[0].append('-s01')
			gaits[0].append(3)
			file_patterns[0].append('-s02')
			gaits[0].append(3)

			# Temporal set
			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N016_train_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[1].append('-n07')
			gaits[1].append(1)
			file_patterns[1].append('-n08')
			gaits[1].append(1)
			file_patterns[1].append('-n09')
			gaits[1].append(1)
			file_patterns[1].append('-n10')
			gaits[1].append(1)
			file_patterns[1].append('-n11')
			gaits[1].append(1)
			file_patterns[1].append('-n12')
			gaits[1].append(1)
			file_patterns[1].append('-b03')
			gaits[1].append(2)
			file_patterns[1].append('-b04')
			gaits[1].append(2)
			file_patterns[1].append('-s03')
			gaits[1].append(3)
			file_patterns[1].append('-s04')
			gaits[1].append(3)
		elif mode == 'ft':
			# Load id file.
			file = open(os.path.join(ids_file_path, 'tumgaidtestids.lst'), mode='r')
			ids = file.read()
			ids = ids[0:-1].split('\n')
			ids = [int(i) for i in ids]

			# Normal set
			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N155_ft_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[0].append('-n01')
			gaits[0].append(1)
			file_patterns[0].append('-n02')
			gaits[0].append(1)
			file_patterns[0].append('-n03')
			gaits[0].append(1)
			file_patterns[0].append('-n04')
			gaits[0].append(1)

			# Temporal set
			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N016_ft_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[1].append('-n07')
			gaits[1].append(1)
			file_patterns[1].append('-n08')
			gaits[1].append(1)
			file_patterns[1].append('-n09')
			gaits[1].append(1)
			file_patterns[1].append('-n10')
			gaits[1].append(1)
		elif mode == 'test':
			# Load id file.
			file = open(os.path.join(ids_file_path, 'tumgaidtestids.lst'), mode='r')
			ids = file.read()
			ids = ids[0:-1].split('\n')
			ids = [int(i) for i in ids]
			set = 3

			# Normal set
			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N155_test' + '_n05-06_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[0].append('-n05')
			gaits[0].append(1)
			file_patterns[0].append('-n06')
			gaits[0].append(1)

			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N155_test' + '_b01-02_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[1].append('-b01')
			gaits[1].append(2)
			file_patterns[1].append('-b02')
			gaits[1].append(2)

			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N155_test' + '_s01-02_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[2].append('-s01')
			gaits[2].append(3)
			file_patterns[2].append('-s02')
			gaits[2].append(3)

			# Temporal set
			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N016_test' + '_n11-12_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[3].append('-n11')
			gaits[3].append(1)
			file_patterns[3].append('-n12')
			gaits[3].append(1)

			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N016_test' + '_b03-04_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[4].append('-b03')
			gaits[4].append(2)
			file_patterns[4].append('-b04')
			gaits[4].append(2)

			file_patterns.append([])
			folders.append('tfimdb_tum_gaid_N016_test' + '_s03-04_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			file_patterns[5].append('-s03')
			gaits[5].append(3)
			file_patterns[5].append('-s04')
			gaits[5].append(3)
		else:
			sys.exit('Unknown mode. Stop')

	elif dataset == 'casiab':
		im_width = 320
		im_height = 240
		subject_pattern = '{:03d}'
		cams = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

		# Build the list of patterns for reading the files. We use a list of lists structure.
		# The first level is used to define a global set of samples and the second level includes all files of that set.
		# For example: training data for tum and tum temporal.
		if mode == 'train':
			ids = np.arange(1, 75, 1)  # First 74 subjects

			# Build set
			file_patterns.append([])
			folders.append('tfimdb_casia_b_N074_train_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			for cam in cams:
				file_patterns[0].append('-nm-01-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-02-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-03-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-04-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-05-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-06-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-bg-01-' + cam)
				gaits[0].append(2)
				file_patterns[0].append('-bg-02-' + cam)
				gaits[0].append(2)
				file_patterns[0].append('-cl-01-' + cam)
				gaits[0].append(3)
				file_patterns[0].append('-cl-02-' + cam)
				gaits[0].append(3)
		elif mode == 'ft':
			ids = np.arange(75, 125, 1)  # Last 50 subjects

			# Normal set
			file_patterns.append([])
			folders.append('tfimdb_casia_b_N050_ft_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			for cam in cams:
				file_patterns[0].append('-nm-01-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-02-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-03-' + cam)
				gaits[0].append(1)
				file_patterns[0].append('-nm-04-' + cam)
				gaits[0].append(1)
		elif mode == 'test':
			ids = np.arange(75, 125, 1)  # Last 50 subjects
			set = 3
			idx = 0
			for cam in cams:
				# Normal set
				file_patterns.append([])
				folders.append('tfimdb_casia_b_N050_test' + '_nm05-06_' + cam + '_' + data_type + str(n_frames) + '_60x60')
				gaits.append([])
				file_patterns[idx].append('-nm-05-' + cam)
				gaits[idx].append(1)
				file_patterns[idx].append('-nm-06-' + cam)
				gaits[idx].append(1)

				file_patterns.append([])
				folders.append('tfimdb_casia_b_N050_test' + '_bg01-02_' + cam + '_' + data_type + str(n_frames) + '_60x60')
				gaits.append([])
				file_patterns[idx+1].append('-bg-01-' + cam)
				gaits[idx+1].append(2)
				file_patterns[idx+1].append('-bg-02-' + cam)
				gaits[idx+1].append(2)

				file_patterns.append([])
				folders.append('tfimdb_casia_b_N050_test' + '_cl01-02_' + cam + '_' + data_type + str(n_frames) + '_60x60')
				gaits.append([])
				file_patterns[idx+2].append('-cl-01-' + cam)
				gaits[idx+2].append(3)
				file_patterns[idx+2].append('-cl-02-' + cam)
				gaits[idx+2].append(3)
				idx = idx + 3
		else:
			sys.exit('Unknown mode. Stop')
	elif dataset == 'ou-mvlp':
		im_width = 1280
		im_height = 960
		subject_pattern = '{:05d}'
		cams = ['000', '015', '030', '045', '060', '075', '090', '180', '195', '210', '225', '240', '255', '270']

		# Build the list of patterns for reading the files. We use a list of lists structure.
		# The first level is used to define a global set of samples and the second level includes all files of that set.
		# For example: training data for tum and tum temporal.
		if mode == 'train':
			file = open(os.path.join(ids_file_path, 'ID_list_train.txt'), mode='r')
			ids = file.read()
			ids = ids[0:-1].split('\n')
			ids = [int(i) for i in ids]

			# Build set
			file_patterns.append([])
			folders.append('tfimdb_ou_mvlp_N05153_train_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			for cam in cams:
				if pattern in '-00-' + cam:
					file_patterns[0].append('-00-' + cam)
					gaits[0].append(1)
				if pattern in '-01-' + cam:
					file_patterns[0].append('-01-' + cam)
					gaits[0].append(1)
		elif mode == 'ft':
			file = open(os.path.join(ids_file_path, 'ID_list_test.txt'), mode='r')
			ids = file.read()
			ids = ids[0:-1].split('\n')
			ids = [int(i) for i in ids]

			# Normal set
			file_patterns.append([])
			folders.append('tfimdb_ou_mvlp_N05154_ft_' + data_type + str(n_frames) + '_60x60')
			gaits.append([])
			for cam in cams:
				file_patterns[0].append('-01-' + cam)
				gaits[0].append(1)
		elif mode == 'test':
			file = open(os.path.join(ids_file_path, 'ID_list_test.txt'), mode='r')
			ids = file.read()
			ids = ids[0:-1].split('\n')
			ids = [int(i) for i in ids]
			set = 3
			idx = 0
			for cam in cams:
				# Normal set
				file_patterns.append([])
				folders.append('tfimdb_ou_mvlp_N05154_test_' + '00_' + cam + '_' + data_type + str(n_frames) + '_60x60')
				gaits.append([])
				file_patterns[idx].append('-00-' + cam)
				gaits[idx].append(1)
				idx = idx + 1
		else:
			sys.exit('Unknown mode. Stop')
	else:
		sys.exit("Unknown dataset. Stop")

	return file_patterns, folders, gaits, set, ids, im_width, im_height, subject_pattern
