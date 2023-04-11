import pickle, logging, numpy as np
from torch.utils.data import Dataset


class Kinetics_Feeder(Dataset):
    def __init__(self, phase, dataset_path, inputs, num_frame, connect_joint, debug, **kwargs):
        self.T = num_frame
        self.inputs = inputs
        self.conn = connect_joint
        data_path = '{}/{}_data.npy'.format(dataset_path, phase)
        label_path = '{}/{}_label.pkl'.format(dataset_path, phase)
        try:
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.name, self.label, self.seq_len = pickle.load(f, encoding='latin1')
        except:
            logging.info('')
            logging.error('Error: Wrong in loading data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()
        if debug:
            self.data = self.data[:300]
            self.label = self.label[:300]
            self.name = self.name[:300]
            self.seq_len = self.seq_len[:300]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.name[idx]
        joint, velocity, bone, motion = self.multi_input(data[:,:self.T,:,:])
        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        if 'M' in self.inputs:
            data_new.append(motion)
        data_new = np.stack(data_new, axis=0)

        return data_new, label, name

    def multi_input(self, data):
        C, T, V, M = data.shape
        joint = np.zeros((C*2, T, V, M))
        velocity = np.zeros((C*2, T, V, M))
        bone = np.zeros((C*2, T, V, M))
        motion = np.zeros((C*2, T, V, M))
        joint[:C,:,:,:] = data
        for i in range(V):
            joint[C:C*2-1,:,i,:] = data[:C-1,:,i,:] - data[:C-1,:,1,:]
            joint[C*2-1,:,i,:] = (data[C-1, :, i, :] + data[C-1, :, 1, :])/2
        for i in range(T-2):
            velocity[:C-1,i,:,:] = data[:C-1,i+1,:,:] - data[:C-1,i,:,:]
            velocity[C-1,i,:,:] = (data[C-1,i+1,:,:] + data[C-1,i,:,:])/2
            velocity[C:C*2-1,i,:,:] = data[:C-1,i+2,:,:] - data[:C-1,i,:,:]
            velocity[C*2-1,i,:,:] = (data[C-1,i+1,:,:] + data[C-1,i,:,:])/2
        for i in range(len(self.conn)):
            bone[:C-1,:,i,:] = data[:C-1,:,i,:] - data[:C-1,:,self.conn[i],:]
            bone[C-1,:,i,:] = (data[C-1,:,i,:] + data[C-1,:,self.conn[i],:])/2
        bone_length = 0
        for i in range(C-1):
            bone_length += bone[i,:,:,:] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        bone[C * 2 - 1, :, :, :] = bone_length
        for i in range(C-1):
            bone[C+i,:,:,:] = np.arccos(bone[i,:,:,:] / bone_length)
        for i in range(T-2):
            motion[:C-1,i,:,:] = bone[:C-1,i+1,:,:] - bone[:C-1,i,:,:]
            motion[C-1,i,:,:] = (bone[C-1,i+1,:,:] + bone[C-1,i,:,:])/2
            motion[C:C*2-1,i,:,:] = bone[C:C*2-1,i+2,:,:] - bone[C:C*2-1,i,:,:]
            motion[C*2-1,i,:,:] = (bone[C*2-1,i+1,:,:] + bone[C*2-1,i,:,:])/2
        return joint, velocity, bone, motion

