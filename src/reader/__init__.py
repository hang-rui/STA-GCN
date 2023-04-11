import logging

from .ntu_reader import NTU_Reader
from .kinetics_reader import Kinetics_Reader


__generator = {
    'ntu': NTU_Reader,
    'kinetics': Kinetics_Reader,
}

def create(args):
    if args.dataset == 'kinetics':
        dataset = args.dataset
        dataset_args = args.dataset_args[dataset]
    else:
        dataset = args.dataset.split('-')[0]
        dataset_args = args.dataset_args[dataset]
    if dataset not in __generator.keys():
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
        raise ValueError()
    return __generator[dataset](args, **dataset_args)
