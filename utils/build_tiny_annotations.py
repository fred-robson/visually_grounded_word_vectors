import os,json



base_fp = os.path.dirname(os.path.abspath(__file__))+"/"
annotations_name = base_fp+"../data/Coco/Annotations/captions_train2014.json" 
tiny_folder = base_fp+"../data/Coco/Tiny2014"
save_name = base_fp+"../data/Coco/Annotations/captions_tiny2014.json" 

tiny_files = os.listdir(tiny_folder)

all_tiny_ids = list()


for t in tiny_files[:-1]: 
	image_id = t[15:-4]
	try: 
		image_id = int(image_id)
		all_tiny_ids.append(image_id)
	except: 
		continue

with open(annotations_name) as f:
	data = json.load(f)

new_annotations = []
found = set()

for d in data["annotations"]:
	d_id = int(d["image_id"])
	if d_id in all_tiny_ids:
		found.add(d_id)
		new_annotations.append(d)

data['annotations'] = new_annotations
print(len(tiny_files))
print(len(all_tiny_ids))
print(len(new_annotations))
print(len(found))

json.dump(data,open(save_name,"w+"))

