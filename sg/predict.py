import pandas as pd
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from dataset import CamvidDataset
from FCN import FCN
import cfg

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

Cam_test = CamvidDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Cam_test, batch_size=1, shuffle=True, num_workers=0)

net = FCN(12).to(device)
net.load_state_dict(t.load("xxx.pth"))
net.eval()

pd_label_color = pd.read_csv('./CamVid/class_dict.csv', sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
	tmp = pd_label_color.iloc[i]
	color = [tmp['r'], tmp['g'], tmp['b']]
	colormap.append(color)

cm = np.array(colormap).astype('uint8')

dir = "./result_pics/"

for i, sample in enumerate(test_data):
	valImg = sample['img'].to(device)
	valLabel = sample['label'].long().to(device)
	out = net(valImg)
	out = F.log_softmax(out, dim=1)
	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
	pre = cm[pre_label]
	pre1 = Image.fromarray(pre)
	pre1.save(dir + str(i) + '.png')
	print('Done')