from turtle import back
import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from openslide import open_slide
import h5py
# from torchvision import summary

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Sampler
from datasets.dataset_h5 import Dataset_All_Bags
from models.resnet_custom import resnet50_baseline
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import cv2
from tqdm import tqdm

from models.model_clam import CLAM_SB, CLAM_MB

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description = 'Arguments for Creating heatmap')
parser.add_argument('--result_dir', type = str, help='directory to store heatmaps')
parser.add_argument('--slides_dir_csv', type = str, help='csv containing Direcotry for Slides')
parser.add_argument('--ckpt_path', type = str, help='Directory for Stored CLAM Model')
parser.add_argument('--h5_dir', type = str, help = 'Directory for h5 files with locations of slides')

parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--n_classes', type=int, default=2, help='Total Number of classes')
parser.add_argument('--csv_path', type=str, help='Path of CSV Containing names of all the slides')
# parser.add_argument('--feat_ext', type = str, help = 'Directory for feature_extractor model')
parser.add_argument('--file_type', type=str, default='.svs', help='Slide Type: svs, tif, etc')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for patches')
parser.add_argument('--patch_size', type=int, default=256, help='Patch size used for creation of features')
parser.add_argument('--downsample', type=int, default=32, help ='Downsampling for heatmaps')
parser.add_argument('--alpha', type=float, default = 0.6, help='Alpha for blending attention score to slide_images')
parser.add_argument('--slides_dir', type = str, help='csv containing Direcotry for Slides')
args = parser.parse_args()


# Create Dataloader:
class patch_loader(Dataset):
	def __init__(self, slide_id, h5_dir):
		path_h5 = os.path.join(h5_dir, slide_id + '.h5')
		with h5py.File(path_h5, 'r') as f:
			self.coords = np.array(f['coords'])
			self.features = torch.from_numpy(np.array(f['features']))
			# print(coords.shape, features.shape)

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		return self.features[idx], self.coords[idx]

## Load CLAM Pretrained Model:
model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
    model_dict.update({"size_arg": args.model_size})
if args.model_type =='clam_sb':
    model = CLAM_SB(**model_dict)
elif args.model_type =='clam_mb':
    model = CLAM_MB(**model_dict)
else: # args.model_type == 'mil'
    if args.n_classes > 2:
        model = MIL_fc_mc(**model_dict)
    else:
        model = MIL_fc(**model_dict)
# model = CLAM_SB
ckpt = torch.load(args.ckpt_path)
ckpt_clean = {}
for key in ckpt.keys():
    if 'instance_loss_fn' in key:
        continue
    ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
model.load_state_dict(ckpt_clean, strict=False)

# model.relocate()
model.to(device)
model.eval()

## Create Result Directory:
if args.result_dir is not None:
	if not os.path.isdir(args.result_dir):
		os.mkdir(args.result_dir)
else:
	print("Error: Please Specify Result directory")
	exit()

soft = nn.Softmax(dim=1) 

slide_paths = pd.read_csv(args.slides_dir_csv)['full_path']
slide_names = pd.read_csv(args.slides_dir_csv)['slide_id']
y_hat = {}
for idx,(s_name, s_path) in enumerate(tqdm(zip(slide_names, slide_paths))):
	
	dataset = patch_loader(s_name, args.h5_dir)
	loader = DataLoader(dataset, batch_size = args.batch_size)
	coords = []
	attention_weights = []
	path_h5 = os.path.join(args.h5_dir, s_name + '.h5')
	with h5py.File(path_h5, 'r') as f:
		coords = np.array(f['coords'])
		features = torch.from_numpy(np.array(f['features']))
	features = features.to(device)
	logits, y_prob, y_h, A_raw, _ = model(features)
	A_soft = soft(A_raw)
	A_soft = A_soft.to('cpu').detach().	numpy().T

	# s_path = os.path.join(args.slides)
	slide = open_slide(os.path.join(s_path))
	# # # slide.read_region()
	m, n = slide.dimensions
	down = args.downsample
	vis_level = slide.get_best_level_for_downsample(down)
	A_soft_log = np.log10(A_soft)
	# print(A_soft.argmin(), A_soft_log.argmin(), A_soft_log.min())
	A_soft_log_norm = (A_soft_log- A_soft_log.min())/(A_soft_log.max() - A_soft_log.min())
	#cmap = plt.get_cmap('jet')
	cmap = plt.get_cmap('coolwarm')
	region_size = (m//down,n//down)
	background = np.array(slide.get_thumbnail((region_size[0],region_size[1])).convert('RGB')).transpose(1,0,2).astype(np.uint8)
	# background = np.zeros((m//down, n//down,3), 'uint8') *255
	attention_img = np.zeros((m//down, n//down,3), 'uint8') *255 
	final_image = background.copy()
	test_img = np.zeros((m//down, n//down), 'uint8') *255
	# Image.fromarray((background).astype(np.uint8)).show()
	# print(attention_img.shape, background.shape, (m,n))
	########below line are commented for tata data #############
	# if float(slide.properties['openslide.mpp-x']) < 0.4:
	# 	args.patch_size = 512
	# else:
	# 	args.patch_size = 256
	# print(float(slide.properties['openslide.mpp-x']) , args.patch_size)
	
	##########till here and added extra line just below of arg.patch_size
	args.patch_size = 512
	for i in range(len(coords)):
		test_img[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down] = np.power(A_soft_log_norm[i],0.7) * 255
		attention_img[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down,:] = (cmap(test_img[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down]))[:,:,:3] * 255
		# Make this from opencv addweighted
		final_image[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down,:] = cv2.addWeighted(attention_img[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down,:],
			args.alpha, background[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down,:], 1 - args.alpha, 0, final_image[coords[i,0]//down:coords[i,0]//down +args.patch_size//down, 
			coords[i,1]//down:coords[i,1]//down + args.patch_size//down,:])
			
	Image.fromarray((final_image.transpose(1,0,2)).astype(np.uint8)).save(os.path.join(args.result_dir, s_name + '_all.png'))
	continue
	