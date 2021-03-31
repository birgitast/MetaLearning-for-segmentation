import torch
from torchmeta.utils.data import BatchMetaDataLoader

from maml import ModelAgnosticMetaLearning
from data import get_datasets
from models import Unet, ResUnet, FCN8
from utils import FocalLoss, BCEDiceFocalLoss, plot_errors, plot_accuracy, plot_iou, DiceLoss

import math, time
import json, os, logging


download_data = True # Download data to local file (won't download if already there)
bce_dice_focal = False # If True, adjusts y_lim in error plot
augment = True # Use data augmentation


#loss_function = torch.nn.BCEWithLogitsLoss()
loss_function = DiceLoss()

"""not working:"""
#loss_function = torch.nn.CrossEntropyLoss()
#loss_function = FocalLoss()
#loss_function = BCEDiceFocalLoss()
#bce_dice_focal = True


def main(args):

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    # Create output folder 
    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        output_folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(output_folder)
        logging.debug('Creating folder `{0}`'.format(output_folder))

        args.datafolder = os.path.abspath(args.datafolder)
        args.model_path = os.path.abspath(os.path.join(output_folder, 'model.th'))


        # Save the configuration in a config.json file
        with open(os.path.join(output_folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(output_folder, 'config.json'))))

    
    


    # Get datasets and load into meta learning format
    meta_train_dataset, meta_val_dataset, _ = get_datasets(args.dataset, args.datafolder, args.num_ways, args.num_shots, args.num_shots_test, augment=augment, fold=args.fold, download=download_data)

    meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    meta_val_dataloader = BatchMetaDataLoader(meta_val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)



 
    # Define model
    model = Unet(device=device, feature_scale=args.feature_scale)  
    model = model.to(device) 
    print(f'Using device: {device}')

    # Define optimizer 
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)#, weight_decay=1e-5)
    #meta_optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum = 0.99)

    # Define meta learner
    metalearner = ModelAgnosticMetaLearning(model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_adaption_steps,
                                            step_size=args.step_size,
                                            learn_step_size=False,
                                            loss_function=loss_function,
                                            device=device)

    best_value = None



    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    train_losses = []
    val_losses = []
    train_ious = []
    train_accuracies = []
    val_accuracies = []
    val_ious = []

    start_time = time.time()

    for epoch in range(args.num_epochs):
        print('start epoch ', epoch+1)
        print('start train---------------------------------------------------')
        train_loss, train_accuracy, train_iou = metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          desc='Training',
                          leave=False)
        print(f'\n train accuracy: {train_accuracy}, train loss: {train_loss}')
        print('end train---------------------------------------------------')
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_ious.append(train_iou)

        # Evaluate in given intervals
        if epoch%args.val_step_size == 0:
            print('start evaluate-------------------------------------------------')
            results = metalearner.evaluate(meta_val_dataloader,
                                            max_batches=args.num_batches,
                                            verbose=args.verbose,
                                            desc=epoch_desc.format(epoch + 1),
                                            is_test=False)
            val_acc = results['accuracy']
            val_loss = results['mean_outer_loss']
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_ious.append(results['iou'])
            print(f'\n validation accuracy: {val_acc}, validation loss: {val_loss}')
            print('end evaluate-------------------------------------------------')

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
                    torch.save(model.state_dict(), f)
        
        print('end epoch ', epoch+1)

    elapsed_time = time.time() - start_time
    print('Finished after ', time.strftime('%H:%M:%S',time.gmtime(elapsed_time)))

    r = {}
    r['train_losses'] = train_losses
    r['train_accuracies'] = train_accuracies
    r['train_ious'] = train_ious
    r['val_losses'] = val_losses
    r['val_accuracies'] = val_accuracies
    r['val_ious'] = val_ious
    r['time'] = time.strftime('%H:%M:%S',time.gmtime(elapsed_time))
    with open(os.path.join(output_folder, 'train_results.json'), 'w') as g:
        json.dump(r, g)
        logging.info('Saving results dict in `{0}`'.format(
                     os.path.abspath(os.path.join(output_folder, 'train_results.json'))))


    # Plot results
    plot_errors(args.num_epochs, train_losses, val_losses, val_step_size=args.val_step_size, output_folder=output_folder, save=True, bce_dice_focal=bce_dice_focal)
    plot_accuracy(args.num_epochs, train_accuracies, val_accuracies, val_step_size=args.val_step_size, output_folder=output_folder, save=True)
    plot_iou(args.num_epochs, train_ious, val_ious, val_step_size=args.val_step_size, output_folder=output_folder, save=True)
    

    if hasattr(meta_train_dataset, 'close'):
        meta_train_dataset.close()
        meta_val_dataset.close()
 





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('datafolder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
        choices=['pascal5i','mydata'], default='pascal5i',
        help='Name of the dataset (default: pascal5i).')
    parser.add_argument('--output-folder', type=str, default='results',
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=1,
        help='Number of classes per task (n in "n-way", default: 1).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')
    parser.add_argument('--fold', type=int, default=0,
        help='The model validates on the given fold (class split) and trains on the other three (default: 0).')


    # Model
    parser.add_argument('--feature-scale', type=int, default=4,
        help='Scaling of number of feature maps.'
        '(default: 4).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-adaption-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true', default=True,
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    parser.add_argument('--val-step-size', type=int, default=5,
        help='Number of epochs after which model is re-evaluated')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)










