import torch
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import pascal5i
from torchmeta.transforms import SegmentationPairTransform, ClassSplitter

from utils import SegmentationPairTransformNorm




def get_datasets(name, folder, num_ways=1, num_shots=1, num_shots_test=None, shuffle=True, seed=None, download=False, fold=0):

    # dataset_transform = ClassSplitter(shuffle=True, num_train_per_class=num_shots, num_test_per_class=num_shots_test))
    transform = SegmentationPairTransformNorm(256) # 252 works, OR set padding in convolution layer = 1 instead of 0 , OR padding=0 and add cropping to conv (like in original Paper)
    if name =='pascal5i':
        dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)

        meta_train_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='train', transform=transform, download=download, fold=fold)#, dataset_transform, class_augmentation)
        meta_val_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='val', transform=transform, download=download, fold=fold)#, dataset_transform, class_augmentation)
        meta_test_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='test', transform=transform, download=download, fold=fold)#, dataset_transform, class_augmentation)

    else: 
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return meta_train_dataset, meta_val_dataset, meta_test_dataset
    
    
    


