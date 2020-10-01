import torch
from torchmeta.utils.data import BatchMetaDataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

from maml import ModelAgnosticMetaLearning
from data import get_datasets, visualize, show_random_data
from models import Unet

import math
import time
import logging
import os
import json
from collections import OrderedDict


output_folder = "results"

# data params
dataset = 'Pascal5i'
download_data = True
path_to_dataset = "/home/birgit/MA/Code/data"
#path_to_dataset = '/no_backups/d1364/data'
num_ways = 1
num_shots = 5
num_shots_test = 15
batch_size = 1
max_batches = 2

# model params
seg_threshold = 0.5
learning_rate = 0.001
first_order = True
num_adaption_steps = 1
step_size = 0.4

use_cuda = False
#use_cuda = True


# training params
num_epochs = 20
verbose = False
num_workers = 8

# test params
config_path = '' # Path to the configuration file returned by `train.py`


#loss_function = torch.nn.NLLLoss()
loss_function = torch.nn.BCEWithLogitsLoss()
#loss_function = torch.nn.CrossEntropyLoss()



def main():

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    device = torch.device('cuda' if use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if (output_folder is not None):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logging.debug('Creating folder `{0}`'.format(output_folder))

        folder = os.path.join(output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        #path_to_dataset = os.path.abspath(path_to_dataset)
        model_path = os.path.abspath(os.path.join(folder, 'model.th'))

        config_dict = {'folder': path_to_dataset, 'dataset': dataset, 'output_folder': output_folder, 'num_ways': num_ways, 'num_shots': num_shots, 
                        'num_shots_test': num_shots_test, 'batch_size': batch_size, 'num_adaption_steps': num_adaption_steps, 'num_epochs': num_epochs,
                        'step_size': step_size, 'first_order': first_order, 'learning_rate': learning_rate, 'num_workers': num_workers, 'verbose': verbose,
                        'use_cuda': use_cuda, 'model_path': model_path}

        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))



    
    # get datasets and load into meta learning format
    meta_train_dataset, meta_val_dataset, meta_test_dataset = get_datasets(dataset, path_to_dataset, num_ways, num_shots, num_shots_test, download=download_data)

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)

    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)
    

    #show_random_data(meta_train_dataset)

    model = Unet()   
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #meta_optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum = 0.99)
    metalearner = ModelAgnosticMetaLearning(model,
                                            meta_optimizer,
                                            first_order=first_order,
                                            num_adaptation_steps=num_adaption_steps,
                                            step_size=step_size,
                                            loss_function=loss_function,
                                            device=device)

    best_value = None

    #dataloader_test(meta_train_dataloader)
    #print('param before training:')
    #print_test_param(model)

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(num_epochs)))
    for epoch in range(num_epochs):
        print("start epoch ", epoch+1)
        metalearner.train(meta_train_dataloader,
                          max_batches=max_batches,
                          verbose=verbose,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=max_batches,
                                       verbose=verbose,
                                       desc=epoch_desc.format(epoch + 1))

        # Save best model
        if 'accuracies_after' in results:
            if (best_value is None) or (best_value < results['accuracies_after']):
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (output_folder is not None):
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
        
        print("end epoch ", epoch+1)

    if hasattr(meta_train_dataset, 'close'):
        meta_train_dataset.close()
        meta_val_dataset.close()
 

    #print('param after training:')
    #print_test_param(model)


    """print('---------------testing--------------------')


    meta_test_dataloader = BatchMetaDataLoader(meta_test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)

    path_to_config = '/home/birgit/MA/Code/torchmeta/gitlab/results/2020-09-28_165404/config.json'

    with open(path_to_config, 'r') as f:
        config = json.load(f)

    test_model = Unet()

    with open(config['model_path'], 'rb') as f:
        test_model.load_state_dict(torch.load(f, map_location=device))

    device = torch.device('cuda' if use_cuda
                        and torch.cuda.is_available() else 'cpu')

    dataloader_test(meta_train_dataloader, test_model)"""



def dataloader_test(dataloader, model=None):
    for batch in dataloader:

        # dataloader test:
        train_inputs, train_targets, train_labels = batch["train"]
        print('Train inputs shape: {0}'.format(train_inputs.shape))    # (batchsize, no of shots, channels = 3 (RGB), h, w)
        print('Train targets shape: {0}'.format(train_targets.shape))  # (batchsize, no of shots, channels = 1, h, w)
        print('Train labels shape: {0}'.format(train_labels.shape))    # (batch size, no of shots)

        test_inputs, test_targets, test_labels = batch["test"]

        label_idx = train_labels[0][0] - 6
        #label = meta_train_dataset.dataset.labels[label_idx]
        label = ''
        print("label ", train_labels)
        visualize(train_inputs[0][0], label + " input")
        visualize(train_targets[0][0], label + " target")

        if model:
            # model test:
            outputs = model(train_inputs[0])
            print('Output shape: {0}'.format(outputs.shape))
            output1 = outputs[0].detach()  
            prob_map = torch.sigmoid(output1)
            mask = prob_map > seg_threshold

            visualize(output1, label + " model output")
            visualize(mask, label + " model mask")
            plt.show()
        break


def print_test_param(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
        break


main()










