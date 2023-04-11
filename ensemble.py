import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', type=str, required=True,
                        choices={'ntu60_xsub', 'ntu60_xview', 'ntu120_xsub', 'ntu120_xset', 'kinetics'},
                        help='Select dataset')
    parser.add_argument('--num_of_stream', '-n', type=int, required=True,
                        choices={2, 3, 4},
                        help='Number of streams for multi stream fusion')
    parser.add_argument('--score_dir', '-dir', type=str,
                        default='./score',
                        help='Directory of score')

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'kinetics' in arg.dataset:
        with open('data/Kinetics_skeleton/npy_dataset/original/kinetics/eval_label.pkl', 'rb') as f:
                name, label, seq_len = pickle.load(f, encoding='latin1')
    elif 'ntu60' in arg.dataset:
        if 'xsub' in arg.dataset:
            with open('datat/NTU_skeleton/npy_dataset/original/ntu-xsub/eval_label.pkl', 'rb') as f:
                name, label, seq_len = pickle.load(f, encoding='latin1')
        elif 'xview' in arg.dataset:
            with open('data/NTU_skeleton/npy_dataset/transformed/ntu-xview/eval_label.pkl', 'rb') as f:
                name, label, seq_len = pickle.load(f, encoding='latin1')
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            with open('data//NTU_skeleton/npy_dataset/original/ntu-xsub120/eval_label.pkl', 'rb') as f:
                name, label, seq_len = pickle.load(f, encoding='latin1')
        elif 'xset' in arg.dataset:
            with open('data/NTU_skeleton/npy_dataset/original/ntu-xset120/eval_label.pkl', 'rb') as f:
                name, label, seq_len = pickle.load(f, encoding='latin1')
    else:
        raise NotImplementedError

    joint_dir = os.path.join(arg.score_dir, "STA-GCN_{}_j_score".format(dataset))
    with open(os.path.join(joint_dir, 'score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    bone_dir = os.path.join(arg.score_dir, "STA-GCN_{}_b_score".format(dataset))
    with open(os.path.join(bone_dir, 'score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.num_of_stream > 2:
        joint_motion_dir = os.path.join(arg.score_dir, "STA-GCN_{}_jm_score".format(dataset))
        with open(os.path.join(joint_motion_dir, 'score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.num_of_stream > 3:
        bone_motion_dir = os.path.join(arg.score_dir, "STA-GCN_{}_bm_score".format(dataset))
        with open(os.path.join(bone_motion_dir, 'score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    if arg.num_of_stream == 4:
        arg.alpha = [0.6, 0.6, 0.4, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.num_of_stream == 3:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Dataset:{}, Multi stream:{}'.format(arg.dataset, arg.num_of_stream))
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
