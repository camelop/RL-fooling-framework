import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np

class CifarModel(object):
    label_set = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def _loadModel(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            checkpoint = torch.load('src/model/mnist/checkpoint/{}'.format(model))        
            self.net.load_state_dict(checkpoint['net'])
        else:
            # original saved file with DataParallel
            state_dict = torch.load('src/model/mnist/checkpoint/{}'.format(model), map_location=lambda storage, loc: storage)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['net'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            self.net.load_state_dict(new_state_dict)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010,))
        ])

    def predict(self, image, prob=False):
        with torch.no_grad():
            if (len(image.shape) == 2):
                image = np.expand_dims(image, axis=-1)
            image = self.transform(image)
            inputs = image[None, :, :, :].to(self.device).float()
            outputs = self.net(inputs)
            outputs = outputs.cpu().numpy()
            if prob:
                return dict(zip(self.label_set, outputs))
            else:
                return self.label_set[np.argmax(outputs)]
        
    def __str__(self):
        return "CifarDefaultModel"