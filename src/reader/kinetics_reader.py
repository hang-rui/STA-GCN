import os, pickle, logging, json, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization


class Kinetics_Reader():
    def __init__(self, args, root_folder, transform, kinetics_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 18
        self.max_person = 5
        self.select_person_num = 2
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)
        self.kinetics_path = kinetics_path

    def read_file(self, file_path):
        skeleton = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.max_person), dtype=np.float32)
        with open(file_path, 'r') as f:
            video_info = json.load(f)
            frame_num = len(video_info['data'])
            label = video_info['label_index']
            for frame_info in video_info['data']:
                frame_index = frame_info['frame_index'] - 1
                for m, skeleton_info in enumerate(frame_info["skeleton"]):
                    if m >= self.max_person:
                        break
                    pose = skeleton_info['pose']
                    score = skeleton_info['score']
                    skeleton[0, frame_index, :, m] = pose[0::2]
                    skeleton[1, frame_index, :, m] = pose[1::2]
                    skeleton[2, frame_index, :, m] = score
        return skeleton[:,:frame_num,:,:], frame_num, label

    def get_nonzero_std(self, s):
        index = s.sum(-1).sum(-1) != 0
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()
        else:
            s = 0
        return s

    def gendata(self, phase):
        if phase == 'eval':
            p = 'val'
        else: p = phase
        folder = '{}/kinetics_{}'.format(self.kinetics_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(self.kinetics_path, p)
        file_list = os.listdir(folder)
        with open(label_path) as f:
            label_info = json.load(f)
        sample_id = [file_name.split('.')[0] for file_name in file_list]
        label = np.array([label_info[id]['label_index'] for id in sample_id])
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        file_list = [s for h, s in zip(has_skeleton, file_list) if h]
        label = label[has_skeleton]

        # phase = train, test
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        iterizer = tqdm(file_list, dynamic_ncols=True) if self.progress_bar else file_list
        for index, file_name in enumerate(iterizer):
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
            # Get sample information
            file_path = os.path.join(folder, file_name)
            # Read one sample
            skeleton, frame_num, file_label = self.read_file(file_path)

            # centralization
            skeleton[0:2] = skeleton[0:2] - 0.5
            skeleton[1:2] = -skeleton[1:2]
            skeleton[0][skeleton[2] == 0] = 0
            skeleton[1][skeleton[2] == 0] = 0

            # get & check label index
            assert (label[index] == file_label)

            # sort by score
            sort_index = (-skeleton[2, :, :, :].sum(axis=1)).argsort(axis=1)
            for t, s in enumerate(sort_index):
                skeleton[:, t, :, :] = skeleton[:, t, :, s].transpose((1, 2, 0))
            skeleton = skeleton[:, :, :, 0:self.select_person_num]
            data[:, :frame_num, :, :] = skeleton

            sample_data.append(data)
            sample_path.append(file_path)
            sample_label.append(file_label)
            sample_length.append(frame_num)
        # Save label
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f)

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # Save data
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)



