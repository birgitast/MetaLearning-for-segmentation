from models import Unet
from utils import visualize
import json, torch
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


from utils import visualize, load_random_sample




def main(args):

    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    mask_path = '/home/birgit/MA/Code/data/pascal5i/Binary_map_aug/train/8'
    jpeg_path = '/home/birgit/MA/Code/data/pascal5i/VOCdevkit/VOC2012/JPEGImages'
    

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder

    model = Unet()

    with open(config['model_path'], 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))


    rnd_img, rnd_mask = load_random_sample(mask_path, jpeg_path)
    

    output = model(rnd_img)
    prob_mask = torch.sigmoid(output)
    prediction = prob_mask.detach()[0] > 0.7

    visualize(rnd_img.squeeze(0), 'input')
    visualize(rnd_mask, 'ground truth')
    visualize(output.detach().squeeze(0), 'output')
    visualize(prediction, 'prediction')

    plt.show()
    #plt.savefig('test.png')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=int, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')
    parser.add_argument('--use-cuda', action='store_true')
    

    args = parser.parse_args()
    main(args)