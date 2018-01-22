# PlacesCNN for scene classification
#
# by Bolei Zhou

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image


# th architecture to use
# arch = 'resnet18'

# load the pre-trained weights
#model_file = 'resnet18_best.pth.tar'
#model_file = torch.load(model_file)
#model=models.__dict__['resnet18'](num_classes=int(80))
#model= torch.nn.DataParallel(model).cuda()
#state_dict=model_file['state_dict']
#model.load_state_dict(state_dict)

#if not os.access(model_file, os.W_OK):
#    weight_url = 'http://places2.csail.mit.edu/models_places365/whole_%s_places365.pth.tar' % arch
#    os.system('wget ' + weight_url)

#useGPU = 1
#if useGPU == 1:
model_file='resnet50_best2.pth.tar'
model = torch.load(model_file)
#else:
#model = torch.load(model_file, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
# from functools import partial
# import pickle
# pickle.load = partial(pickle.load, encoding="latin1")
# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
# model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)


model.eval()

# load the image transformer
centre_crop = trn.Compose([
        trn.Scale(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label

#if not os.access(file_name, os.W_OK):
#    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#    os.system('wget ' + synset_url)
#classes = list()
#for a in range(0,80):
#	classes.append(str(a))
#classes = tuple(classes)

# load the test image
img_name = '002.jpg'
#img_url = 'http://places.csail.mit.edu/demo/' + img_name
#os.system('wget ' + img_url)
img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
print list(idx[0:3])
print 'RESULT ON ' 
# output the prediction
for i in range(0, 3):
    print('{:.3f} -> {}'.format(probs[i], idx[i]))
