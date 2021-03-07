import torch
import torch.nn.functional as F
from torchmeta.utils.data import BatchMetaDataLoader

import json, tkinter, os


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


from models import Unet
from utils import print_test_param, DiceLoss, dataloader_test

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


    dataset = 'pascal5i'
    if 'fold' in config.keys():
        fold = config['fold']
    else:
        fold=0
    #dataset = 'mydata'
    #fold=[7]

    padding = 1
    """Not needed with transform = SegmentationPairTransformNorm(256)"""
    """elif dataset=='mydata':
        padding=2"""


    if 'feature_scale' in config.keys():
        model = Unet(feature_scale=config['feature_scale'], padding=padding)
    else:
        model = Unet(feature_scale=4, padding=padding)

    
    
    print('fold: ', fold)

    #print_test_param(model)
    # get datasets and load into meta learning format
    meta_train_dataset, meta_val_dataset, meta_test_dataset = get_datasets(dataset, data_path, config['num_ways'], config['num_shots'], config['num_shots_test'], fold=fold, download=False, augment=False)


    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    
    print('num shots = ', config['num_shots'])
    print(f'Using device: {device}')

    #meta_optimizer = torch.optim.Adam(model.parameters(), lr=config['meta_lr'])



    with open(config['model_path'], 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    
    #model.train(False)


    metalearner = ModelAgnosticMetaLearning(model,
                                            first_order=config['first_order'],
                                            num_adaptation_steps=config['num_adaption_steps'],
                                            step_size=config['step_size'],
                                            loss_function=loss_function,
                                            device=device)

    results = metalearner.evaluate(meta_val_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   desc='Test',
                                   is_test=True)
    


    if dataset=='pascal5i':
        labels =['aeroplane', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        accuracies = [value for _, value in results['mean_acc_per_label'].items()]
        ious = [value for _, value in results['mean_iou_per_label'].items()]
        
        val_ious = [x for x in ious if x>0.0]
        val_accs = [x for x in accuracies if x>0.0]
        """print(results['mean_iou_per_label'])
        print('macc: ', sum(val_accs)/len(val_accs))
        print('mIoU: ', sum(val_ious)/len(val_ious))
        print(results['mean_acc_per_label'])
        print(results['mean_iou_per_label'])"""




        y_pos = np.arange(len(labels))
        

        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.barh(y_pos, accuracies, align='center', alpha=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('acc')
        ax1.set_xlim(0, 1)
        ax1.set_title('Accuracies per label')

        ax2.barh(y_pos, ious, align='center', alpha=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('iou')
        ax2.set_xlim(0, 1)
        ax2.set_title('IoU scores per label')
        plt.grid(True)


        plt.show()


        #print_test_param(model)

        """for batch in meta_val_dataloader:
            _, _, train_labels = batch['train']
            #label = train_labels
            label = train_labels[0][0].item()
            print('label: ', label)
            break"""
        
    
    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'test_results.json'), 'w') as f:
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

