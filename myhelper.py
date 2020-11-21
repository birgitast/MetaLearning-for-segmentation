import warnings

#from torchmeta.datasets import (Omniglot, MiniImagenet, TieredImagenet, CIFARFS,
                                #CUB, DoubleMNIST, TripleMNIST, Pascal5i)
from mypascal5i import Pascal5i
from torchmeta.transforms import Categorical, ClassSplitter, Rotation, SegmentationPairTransform
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

__all__ = [
    'omniglot',
    'miniimagenet',
    'tieredimagenet',
    'cifar_fs',
    'cub',
    'doublemnist',
    'triplemnist'
]

def helper_with_default(klass, folder, shots, ways, shuffle=True,
                        test_shots=None, seed=None, defaults={}, **kwargs):

    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
            'set in the helper function for the number of classes per task. '
            'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', ToTensor())
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = defaults.get('target_transform',
                                                  Categorical(ways))
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)
    if test_shots is None:
        test_shots = shots
    dataset = klass(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset



def pascal5i(folder, shots, ways=1, shuffle=True, test_shots=None,
             seed=None, **kwargs):
    """Helper function to create a meta-dataset for the PASCAL-VOC dataset.
    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.
    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.
    ways : int
        Number of classes per task. This corresponds to `N` in `N-way`
        classification. Only supports 1-way currently
    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.
    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.
    seed : int, optional
        Random seed to be used in the meta-dataset.
    kwargs
        Additional arguments passed to the `Omniglot` class.
    """
    defaults = {
        'transform': SegmentationPairTransform(500),
        'class_augmentations': []
    }
    return helper_with_default(Pascal5i, folder, shots, ways,
                               shuffle=shuffle, test_shots=test_shots,
                               seed=seed, defaults=defaults, **kwargs)