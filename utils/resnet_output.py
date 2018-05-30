import torch 
import torchvision.models as models
import torchvision.transforms as transforms
from data_utils import CocoCaptions,FlickrCaptions
import numpy as np
from tqdm import tqdm
import os

broken_images_file = "saved_items/broken_images.txt"
image_size = 224 #Size of smallest dimension

'''
All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W
are expected to be at least 224. The images have to be loaded in to a range of [0, 1] 
and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
You can use the following transform to normalize:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
'''

def main(Captions,ignore_prev = False):
	'''
	Creates output for data loaded from coco. @data refers to what CocoCaptions type to load 
	'''

	Model = models.resnet101(pretrained = True).eval()
	Model = Model.float()

	

	for image,image_id in tqdm(Captions.get_all_images(),total=Captions.num_images()):
		
		save_address = Captions.get_image_resnet_address(image_id)
		if os.path.isfile(save_address) and ignore_prev: continue 

		#Convert to PiL image for resizing + croppping 
		convert_to_pil = transforms.ToPILImage()
		image = convert_to_pil(image)

		image = transforms.functional.resize(image,size=image_size) #model requires everything is at least 

		trans = transforms.Compose([			
									transforms.RandomCrop(size=image_size),
									transforms.ToTensor(), #Convert back to tensor to normalize
									transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
									])
		
		image = trans(image)
		image = image.unsqueeze(0) #unsqueeze bc it needs to be a batch. Here use a batch of size 1
		output = Model(image)
		output_np = output.detach().numpy()
		np.save(save_address,output_np)
		



if __name__ == "__main__":
	main(FlickrCaptions(0),True)
	main(FlickrCaptions(1),True)
	main(FlickrCaptions(3),True)
	#main(2,True)








