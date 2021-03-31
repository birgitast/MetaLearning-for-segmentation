import torch

import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from collections import OrderedDict
from torchmeta.modules import MetaModule

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import PIL


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


class SegmentationPairTransformNorm(object):
    """Transform both the image and its respective mask"""
    # Normalization: imagenet normalization
    def __init__(self, target_size):
        self.image_transform = Compose([Resize((target_size, target_size)), ToTensor()])#, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.mask_transform = Compose([Resize((target_size, target_size),
                                               interpolation=PIL.Image.NEAREST), ToTensor()])#, Normalize([0.5], [0.5])])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask



def print_test_param(model):
    """Output one model parameter and its value"""
    for name, param in model.named_parameters():
        #if param.requires_grad:
        print(name, param.data)
        break

        
def plot_accuracy(num_epochs, train_acc, val_acc, val_step_size, output_folder, save=True):
    """Plot training and validation accuracy over the given epochs"""
    plt.plot(range(0, num_epochs), train_acc, 'r--', label='Training Accuracy')
    plt.plot(range(0, num_epochs, val_step_size), val_acc, 'b-', label='Validation Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.grid(True)
    #plt.show()
    if save:
        plt.savefig(output_folder + '/accuracies.png')
        plt.clf()

def plot_errors(num_epochs, train_losses, val_losses, val_step_size, output_folder, save=True, bce_dice_focal=False):
    """Plot training and validation error over the given epochs"""
    import matplotlib.pyplot as plt
    plt.plot(range(0, num_epochs), train_losses, 'r--', label='Training Loss')
    plt.plot(range(0, num_epochs, val_step_size), val_losses, 'b-', label='Validation Loss')
    if bce_dice_focal:
        plt.ylim(bottom=0)        
    else:
        plt.ylim(0, 1)
    plt.xlim(left=0)
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss vs Validation Loss')
    plt.grid(True)
    #plt.show()
    if save:
        plt.savefig(output_folder + '/losses.png')
        plt.clf()

def plot_iou(num_epochs, train_iou, val_iou, val_step_size, output_folder, save=True):
    """Plot training and validation Intersection over Union score over given epochs"""
    plt.plot(range(0, num_epochs), train_iou, 'r--', label='Training IoU')
    plt.plot(range(0, num_epochs, val_step_size), val_iou, 'b-', label='Validation IoU')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('IoU')
    plt.title('Training IoU vs Validation IoU')
    plt.grid(True)
    #plt.show()
    if save:
        plt.savefig(output_folder + '/iou.png')
        plt.clf()



def visualize(tensor, class_name=None):
    """Plot a given image tensor or list of tensors"""
    if type(tensor)!=list:
        img = tensor.permute(1, 2, 0)
        img = img.numpy()
        plt.figure()
        plt.title(class_name)
        plt.imshow(img)

    else:    
        fig, axes = plt.subplots(1, len(tensor))
        for i in range(len(tensor)):
            img = tensor[i].permute(1, 2, 0)
            img = img.numpy()
            ax = plt.gca()
            axes[i].xaxis.set_visible(False)
            axes[i].yaxis.set_visible(False)
            axes[i].imshow(img)
            



def show_random_data(meta_train_dataset):
    """Plot one random image + corresponding mask from training dataset"""
    # Get data by class, e.g. data_by_class[0]: all bus tuples; im, mask, label_idx = meta_train_dataset.dataset[0][0]
    classes = meta_train_dataset.dataset.labels
    data_by_class = meta_train_dataset.dataset 

    # Choose a random image + mask pait out of a random class
    rnd_class_idx = random.randint(0, len(classes)-1)
    rnd_class = classes[rnd_class_idx]
    rnd_im, rnd_mask, _ = random.choice(data_by_class[rnd_class_idx])
    visualize([rnd_im, rnd_mask], rnd_class)
    #plt.show()
    plt.save('random.png')




def get_dice_score(pred, targets, smooth=1):
    """Return the dice score of a given prediction and the ground truth"""
    # Comment out if your model contains a sigmoid or equivalent activation layer
    pred = torch.sigmoid(pred)
    # hard dice score:
    #pred = (pred > 0.5).float()       
        
    # Flatten label and prediction tensors
    pred = pred.view(-1)
    targets = targets.view(-1)
        
    intersection = (pred * targets).sum()                            
    dice = (2.*intersection + smooth)/(pred.sum() + targets.sum() + smooth) 

    return dice



# from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
def jaccard_idx(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]

    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1).type(torch.LongTensor)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_idx = (intersection / (union + eps)).mean()
    return jacc_idx


class DiceLoss(torch.nn.Module):
    """Compute the Dice Loss between an image and target out of the dice score"""
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        dice_score = get_dice_score(inputs, targets, smooth)
        
        return 1 - dice_score
        





""" --------------------------------------- ALTERNATIVE LOSS FUNCTIONS -----------------------------------------------------"""

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

from torch.autograd import Variable


# 1) from https://github.com/achaiah/pywick/blob/master/pywick/losses.py

class FocalLoss(nn.Module):
    """
    Weighs the contribution of each sample to the loss based in the classification error.
    If a sample is already classified correctly by the CNN, its contribution to the loss decreases.
    :eps: Focusing parameter. eps=0 is equivalent to BCE_loss
    """
    def __init__(self, l=0.5, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.l = l
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.view(-1)
        probs = torch.sigmoid(logits).view(-1)

        losses = -(targets * torch.pow((1. - probs), self.l) * torch.log(probs + self.eps) + \
                   (1. - targets) * torch.pow(probs, self.l) * torch.log(1. - probs + self.eps))
        loss = torch.mean(losses)

        return loss




# 2) from https://github.com/achaiah/pywick/blob/master/pywick/losses.py
class BCEDiceFocalLoss(nn.Module):
    '''
        :param num_classes: number of classes
        :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                            focus on hard misclassified example
        :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
        :param weights: (list(), default = [1,1,1]) Optional weighing (0.0-1.0) of the losses in order of [bce, dice, focal]
    '''
    def __init__(self, focal_param=0.5, weights=[1.0,1.0,1.0], **kwargs):
        super(BCEDiceFocalLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(l=focal_param)
        self.weights = weights

    def forward(self, logits, targets):
        logits = logits.squeeze()
        targets = torch.squeeze(targets, dim=1)
        return self.weights[0] * self.bce(logits, targets) + self.weights[1] * self.dice(logits, targets) + self.weights[2] * self.focal(logits.unsqueeze(1), targets.unsqueeze(1))


