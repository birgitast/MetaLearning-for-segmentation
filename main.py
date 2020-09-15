import torch
from torchmeta.utils.data import BatchMetaDataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt

from metalearners import ModelAgnosticMetaLearning
from data import get_datasets, visualize, show_random_data
from models import Unet

import math
from collections import OrderedDict


# data params
dataset = 'Pascal5i'
path_to_dataset = "/home/birgit/MA/Code/data"
ways = 1
shots = 5
batch_size = 16

# model params
seg_threshold = 0.5
learning_rate = 0.001
first_order = True
num_adaption_steps = 1
step_size = 0.1
device = "cpu"


# training params
num_epochs = 1
num_batches = 2
verbose = True 
num_workers = 8

loss_function = F.cross_entropy


def main():

    # get datasets and load into meta learning format
    meta_train_dataset, meta_val_dataset, meta_test_dataset = get_datasets(dataset, path_to_dataset, ways, shots)
    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)
    

    show_random_data(meta_train_dataset)

    model = Unet()    

    
    for batch in meta_train_dataloader:

        # dataloader test:
        train_inputs, train_targets, train_labels = batch["train"]
        print('Train inputs shape: {0}'.format(train_inputs.shape))    # (batchsize, no of shots, channels = 3 (RGB), h, w)
        print('Train targets shape: {0}'.format(train_targets.shape))  # (batchsize, no of shots, channels = 1, h, w)
        print('Train labels shape: {0}'.format(train_labels.shape))    # (batch size, no of shots)

        test_inputs, test_targets, test_labels = batch["test"]

        # model test:
        outputs = model(train_inputs[0])
        print('Output shape: {0}'.format(outputs.shape))
        output1 = outputs[0].detach()  
        prob_map = torch.sigmoid(output1)
        mask = prob_map > seg_threshold

        label_idx = train_labels[0][0] - 6
        label = meta_train_dataset.dataset.labels[label_idx]

        visualize(train_inputs[0][0], label + " input")
        visualize(train_targets[0][0], label + " target")
        visualize(output1, label + " model output")
        visualize(mask, label + " model mask")
        plt.show()

        break


    
    
    # TODO: Training (just copied from https://github.com/tristandeleu/pytorch-meta)

    """meta_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metalearner = ModelAgnosticMetaLearning(model,
                                            meta_optimizer,
                                            first_order=first_order,
                                            num_adaptation_steps=num_adaption_steps,
                                            step_size=step_size,
                                            loss_function=loss_function,
                                            device=device)

    best_value = None
    prob_map = torch.sigmoid(output[0]).detach()


    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(num_epochs)))
    for epoch in range(num_epochs):
        metalearner.train(meta_train_dataloader,
                          max_batches=num_batches,
                          verbose=verbose,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=num_batches,
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

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)

    if hasattr(benchmark.meta_train_dataset, 'close'):
        benchmark.meta_train_dataset.close()
        benchmark.meta_val_dataset.close()"""



main()









