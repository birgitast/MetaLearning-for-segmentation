import torch
from myhelper import pascal5i
import pairtransforms as pt
#from torchmeta.transforms import SegmentationPairTransform
from utils import SegmentationPairTransformNorm



def get_datasets(name, folder, num_ways=1, num_shots=1, num_shots_test=None, shuffle=True, seed=None, download=False, fold=0, augment=False):
    """Store data in a meta-learning format, and return it split into training, validation and test dataset"""
    if augment:
        # Add desired augmentation techniques
        augmentations = [
                        pt.RandomHorizontalFlip(p=0.5),
                        pt.RandomVerticalFlip(p=0.5),                  
                        pt.ColorJitter(brightness=1, contrast=0.6, saturation=(0.5, 0.5), hue=[-0.2, 0.2]),
                        #pt.RandomCrop(size=32, padding=4),
                        pt.RandomRotation(degrees=45)
                        ]
    else:
        augmentations = None

    
    if name == 'pascal5i':

        transform = SegmentationPairTransformNorm(256) # 252 works, OR set padding in convolution layer = 1 instead of 0 , OR padding=0 and add cropping to conv (like in original U-Net Paper)

        meta_train_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='train', transform=transform, download=download, class_augmentations=augmentations, fold=fold)
        meta_val_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='val', transform=transform, download=download, fold=fold)
        meta_test_dataset = None


    else: 
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return meta_train_dataset, meta_val_dataset, meta_test_dataset