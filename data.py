import torch
from torchmeta.utils.data import BatchMetaDataLoader
#from torchmeta.datasets.helpers import pascal5i
from myhelper import pascal5i
from torchmeta.transforms import SegmentationPairTransform, ClassSplitter, Rotation, DefaultTargetTransform
import pairtransforms as pt
from torchvision import transforms

from utils import SegmentationPairTransformNorm



def get_datasets(name, folder, num_ways=1, num_shots=1, num_shots_test=None, shuffle=True, seed=None, download=False, fold=0, augment=False):

    dataset_transform = ClassSplitter(shuffle=True, num_train_per_class=num_shots, num_test_per_class=num_shots_test)

    if augment:
        augmentations = [
                        pt.RandomHorizontalFlip(p=0.5),
                        pt.RandomVerticalFlip(p=0.5),                  
                        pt.ColorJitter(brightness=1, contrast=0.6, saturation=(0.5, 0.5), hue=[-0.2, 0.2]),
                        #pt.RandomCrop(size=32, padding=4)
                        #pt.GaussianBlur(kernel_size=5),
                        #pt.RandomGrayscale(p=0.1),
                        pt.RandomRotation(degrees=45)
                        ]
    else:
        augmentations = None

    
    transform = SegmentationPairTransformNorm(256) # 252 works, OR set padding in convolution layer = 1 instead of 0 , OR padding=0 and add cropping to conv (like in original Paper)
    
    if name == 'pascal5i':

        meta_train_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='train', transform=transform, download=download, class_augmentations=augmentations, fold=fold)#, dataset_transform=dataset_transform)
        meta_val_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='val', transform=transform, download=download, class_augmentations=augmentations,fold=fold)#, dataset_transform=dataset_transform)
        meta_test_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='test', transform=transform, download=download, class_augmentations=augmentations, fold=fold)#, dataset_transform=dataset_transform)

    else: 
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return meta_train_dataset, meta_val_dataset, meta_test_dataset