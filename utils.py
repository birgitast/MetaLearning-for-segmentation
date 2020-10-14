import torch

import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from collections import OrderedDict
from torchmeta.modules import MetaModule


def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.
    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import PIL


class SegmentationPairTransformNorm(object):
    # normalization: imagenet normalization! may need to be adjusted, maybe Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    def __init__(self, target_size):
        self.image_transform = Compose([Resize((target_size, target_size)), ToTensor()])#, Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_transform = Compose([Resize((target_size, target_size),
                                               interpolation=PIL.Image.NEAREST),
                                       ToTensor()])#, Normalize([0.5], [0.5])])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask



def dataloader_test(dataloader, model=None):
    for batch in dataloader:

        # dataloader test:
        train_inputs, train_targets, train_labels = batch['train']
        print('Train inputs shape: {0}'.format(train_inputs.shape))    # (batchsize, no of shots, channels = 3 (RGB), h, w)
        print('Train targets shape: {0}'.format(train_targets.shape))  # (batchsize, no of shots, channels = 1, h, w)
        print('Train labels shape: {0}'.format(train_labels.shape))    # (batch size, no of shots)

        test_inputs, test_targets, test_labels = batch['test']

        label_idx = train_labels[0][0] - 6
        #label = meta_train_dataset.dataset.labels[label_idx]
        label = ''
        print('label ', train_labels)
        visualize(train_inputs[0][0], label + ' input')
        visualize(train_targets[0][0], label + ' target')

        if model:
            # model test:
            outputs = model(train_inputs[0])
            print('Output shape: {0}'.format(outputs.shape))
            output1 = outputs[0].detach()  
            prob_map = torch.sigmoid(output1)
            mask = (prob_map > seg_threshold).float()

            visualize(output1, label + ' model output')
            visualize(mask, label + ' model mask')
            plt.show()
        break


def print_test_param(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
        break


def plot_errors(num_epochs, train_losses, val_losses, val_step_size, output_folder, save=True):
    plt.plot(range(0, num_epochs), train_losses, 'r--', label='Training Loss')
    plt.plot(range(0, num_epochs, val_step_size), val_losses, 'b-', label='Validation Loss')
    plt.ylim(0, None)
    plt.legend(loc='upper right')
    plt.title('Training Loss vs Validation Loss')
    #plt.show()
    if save:
        plt.savefig(output_folder + '/losses.png')



# plot a given image tensor
def visualize(tensor, class_name=None):

    img = tensor.permute(1, 2, 0)
    img = img.numpy()

    plt.figure()
    plt.title(class_name)
    plt.imshow(img)



# plot one random image + corresponding mask from training data
def show_random_data(meta_train_dataset):

    classes = meta_train_dataset.dataset.labels
    data_by_class = meta_train_dataset.dataset # eg. data_by_class[0]: all bus tuples; im, mask, label_idx = meta_train_dataset.dataset[0][0]

    rnd_class_idx = random.randint(0, len(classes)-1)
    rnd_class = classes[rnd_class_idx]
    rnd_im, rnd_mask, _ = random.choice(data_by_class[rnd_class_idx])
    visualize(rnd_im, rnd_class)
    visualize(rnd_mask, rnd_class)

    plt.show()



def load_random_sample(mask_path, jpeg_path):
    import os
    
    rnd_name = random.choice([x for x in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, x))])
    rnd_mask = Image.open(mask_path + '/' + rnd_name)
    rnd_img = Image.open(jpeg_path + '/' + rnd_name[0:11]+'.jpg')
    tensor_transform = SegmentationPairTransformNorm(256)
    img, mask = tensor_transform(rnd_img, rnd_mask)
    img = torch.unsqueeze(img, 0) 

    return img, mask