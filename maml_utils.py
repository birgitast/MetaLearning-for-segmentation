import torch

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

