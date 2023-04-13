from PIL import ImageFile
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from skimage.util import random_noise
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models import vgg13, VGG13_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg13_bn, VGG13_BN_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights


def contrast(x, per_size, num):
    c = np.random.uniform(1-per_size, 1-per_size/2, (num, 1, 1, 1))
    x = x.repeat(num, 1, 1, 1).cpu()
    means = np.mean(x.numpy(), axis=(2, 3), keepdims=True)
    return np.clip((x[0] - means[0]) * c + means, 0, 1).float()


def shot_noise(x, per_size, num):
    x = x.repeat(num, 1, 1, 1)
    return torch.clamp(torch.poisson(x * per_size) / per_size, 0, 1)


def impulse_noise(x, per_size, num):
    x = x.repeat(num, 1, 1, 1).cpu()
    for i in range(num):
        x[i] = torch.tensor(random_noise(np.array(x[i]), mode='s&p', amount=per_size))
    return torch.clamp(x, 0, 1)


ImageFile.LOAD_TRUNCATED_IMAGES = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = [shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
       resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1),
       vgg13(weights=VGG13_Weights.IMAGENET1K_V1), vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1),
       vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1), vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
       mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1), vgg19(weights=VGG19_Weights.IMAGENET1K_V1),
       regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V1), resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
       vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1), mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)]

net_num = len(net)

for i in range(net_num):
    net[i] = net[i].to(device)
    if device == 'cuda':
        net[i] = torch.nn.DataParallel(net[i])
        cudnn.benchmark = True
    net[i].eval()

transform_test = transforms.Compose([
    transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(224),
])
val_dataset = torchvision.datasets.ImageFolder(root='../../../dataset/ILSVRC2012-val', transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
dataiter = iter(val_loader)
round_num = 10
result_density = [[0 for i in range(net_num)] for j in range(round_num)]
for ii in range(round_num):
    while True:
        images, labels = dataiter.next()
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            images_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images)
            all_correct = True
            for i in range(net_num):
                if net[i](images_norm).max(1)[1][0] != labels:
                    all_correct = False
                    break
            if all_correct:
                break

    for i in range(net_num):
        per_par = 0.06
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        per_img = transform(impulse_noise(images, per_par, 1))
        outputs = net[i](per_img).max(1)[1]
        if outputs == labels:
            result_density[ii][i] = 1
        else:
            result_density[ii][i] = 0

print(result_density)
