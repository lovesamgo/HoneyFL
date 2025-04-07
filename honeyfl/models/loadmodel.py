import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from transformers import AutoModelForImageClassification
from torchvision.models import resnet18, resnet50
from honeyfl.data.dataload import ImageDataLoader
import torch
from torch.utils.data import DataLoader

from honeyfl.models.Nets import CNNMnist


class HFModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        return output.logits
    

class Model_loader():

    def __init__(self, dataset):
        self.dataset = dataset

    def load_model(self):
        if self.dataset == 'mnist':
            model = CNNMnist()
        elif self.dataset == 'cifar':
            model = resnet18(num_classes=10)
            checkpoint = torch.load(os.path.join(os.path.dirname(__file__), 
                                                'resnet18_cifar/resnet18-cifar10.ckpt'))
            model.load_state_dict(checkpoint)
        elif self.dataset == 'gtsrb':
            model = resnet50(num_classes=43)
        elif self.dataset == 'imagenet':
             base_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50", 
                                                            cache_dir=os.path.join(os.path.dirname(__file__), 'resnet50_imagenet'), 
                                                            num_labels =100,
                                                            ignore_mismatched_sizes=True)
             model = HFModelWrapper(base_model)
        else:
            exit('Error: unrecognized model')

        return model


if __name__ == '__main__':

    model_loader = Model_loader('imagenet')
    model = model_loader.load_model().to('cuda')
    dataset = ImageDataLoader('imagenet')
    train, test = dataset.load_data()
    data_loader = DataLoader(train, batch_size=1, shuffle=True)
    model.eval()
    for images, lable in data_loader:
        with torch.no_grad():
            image, label = images.to('cuda'), lable.to('cuda')
            logits = model(image)
            pros = torch.nn.functional.softmax(logits, dim=-1)
            ground_truth = lable
            print(pros.argmax(-1).item(), ground_truth)
        break
        