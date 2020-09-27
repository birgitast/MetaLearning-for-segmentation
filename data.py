import torch
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import pascal5i
from torchmeta.transforms import SegmentationPairTransform, ClassSplitter

import matplotlib.pyplot as plt
import numpy as np
import random




def get_datasets(name, folder, num_ways=1, num_shots=1, num_shots_test=None, shuffle=True, seed=None, download=False):

    # dataset_transform = ClassSplitter(shuffle=True, num_train_per_class=num_shots, num_test_per_class=num_shots_test))
    transform = SegmentationPairTransform(256) # 252 works, OR set padding in convolution layer = 1 instead of 0 , OR padding=0 and add cropping to conv (like in original Paper)
    if name =='Pascal5i':
        dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)

        meta_train_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='train', transform=transform, download=download)#, dataset_transform, class_augmentation, fold)
        meta_val_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='val', transform=transform, download=download)#, dataset_transform, class_augmentation, fold)
        meta_test_dataset = pascal5i(folder, ways=num_ways, shots=num_shots, test_shots=num_shots_test, 
                                        meta_split='test', transform=transform, download=download)#, dataset_transform, class_augmentation, fold)

    else: 
        print("not implemented")

    return meta_train_dataset, meta_val_dataset, meta_test_dataset





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