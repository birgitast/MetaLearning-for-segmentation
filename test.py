import torch
import torch.nn.functional as F
from torchmeta.utils.data import BatchMetaDataLoader

import json, tkinter, os


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from models import Unet
from utils import visualize, load_random_samples, print_test_param, DiceLoss, dataloader_test

from data import get_datasets
from maml import ModelAgnosticMetaLearning


def main(args):

    data_path='/home/birgit/MA/Code/torchmeta/gitlab/data'
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches

    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    loss_function = DiceLoss()
    fold=1
    threshold=0.6

    if 'feature_scale' in config.keys():
        model = Unet(feature_scale=config['feature_scale'])
    else:
        model = Unet(feature_scale=4)

    #print_test_param(model)
    # get datasets and load into meta learning format
    meta_train_dataset, meta_val_dataset, meta_test_dataset = get_datasets('pascal5i', data_path, config['num_ways'], config['num_shots'], config['num_shots_test'], fold=fold, download=False)

    print('bath size = ',config['batch_size'])

    meta_test_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    
    print('num shots = ', config['num_shots'])
    print(f'Using device: {device}')

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=config['meta_lr'])



    with open(config['model_path'], 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))


    metalearner = ModelAgnosticMetaLearning(model,
                                            first_order=config['first_order'],
                                            num_adaptation_steps=config['num_adaption_steps'],
                                            step_size=config['step_size'],
                                            loss_function=loss_function,
                                            device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   desc='Test',
                                   is_test=True)
    
    print('results: ', results)

    labels =['aeroplane', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    accuracies = [value for _, value in results['mean_acc_per_label'].items()]


    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np

    y_pos = np.arange(len(labels))

    plt.barh(y_pos, accuracies, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xlabel('acc')
    plt.xlim(0, 1)
    plt.title('Accuracies per label')

    plt.show()


    #print_test_param(model)

    """for batch in meta_test_dataloader:
        _, _, train_labels = batch['train']
        #label = train_labels
        label = train_labels[0][0].item()
        print('label: ', label)
        break"""
        
    
    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'results.json'), 'w') as f:
        json.dump(results, f)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=int, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')

    # Optimization
    parser.add_argument('--num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--use-cuda', action='store_true', default=True)

    args = parser.parse_args()
    main(args)

