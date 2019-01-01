import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np

class MnistModelBase(object):
    label_set = [str(i) for i in range(10)]

    def _loadModel(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = self.net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        checkpoint = torch.load('src/model/mnist/checkpoint/{}'.format(model))        
        self.net.load_state_dict(checkpoint['net'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
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