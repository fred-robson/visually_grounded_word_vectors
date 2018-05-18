import torch 
import torchvision.models as models
import torchvision.transforms as transforms
from data_utils import CocoCaptions
import numpy as np

Model = models.resnet101(pretrained = True).eval()
Model = Model.double()


coco = CocoCaptions(3)

for image,image_id in coco.get_all_images():
	

	image = image.astype(float,copy=False)
	image = np.moveaxis(image,2,0) #i.e. mini-batches of 3-channel RGB images of shape (3 x H x W)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
	image = torch.from_numpy(image)
	normalize(image)
	image = image.unsqueeze(0)
	print(image.shape)
	output = Model(image)








'''

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W
are expected to be at least 224. The images have to be loaded in to a range of [0, 1] 
and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
You can use the following transform to normalize:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
'''



