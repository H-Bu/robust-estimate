import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2
from skimage.filters import gaussian


def contrast(x, per_size, num):
    c = np.random.uniform(1-per_size, 1-per_size/2, (num, 1, 1, 1))
    x = x.repeat(num, 1, 1, 1).cpu()
    means = np.mean(x.numpy(), axis=(2, 3), keepdims=True)
    return np.clip((x[0] - means[0]) * c + means, 0, 1).float()


def gauss_noise(x, per_size, num):
    x = x.repeat(num, 1, 1, 1)  # 并行
    noise = (torch.randn(size=x.shape)*per_size).to(device)
    return torch.clamp(x + noise, 0, 1)


def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x, per_size, num):
    x = x.repeat(num, 1, 1, 1).cpu()
    max_val = x.max()
    for i in range(num):
        x[i] += per_size[0] * plasma_fractal(wibbledecay=per_size[1])[:32, :32][np.newaxis, ...]
    return np.clip(x * max_val / (max_val + per_size[0]), 0, 1).float()


def glass_blur(x, per_size, num):
    x = x.repeat(num, 1, 1, 1).cpu()
    for j in range(num):
        x[j] = torch.from_numpy(gaussian(np.array(x[j]), sigma=per_size[0], channel_axis=0))
        # locally shuffle pixels
        for i in range(per_size[2]):
            for h in range(32 - per_size[1], per_size[1], -1):
                for w in range(32 - per_size[1], per_size[1], -1):
                    dx, dy = np.random.randint(-per_size[1], per_size[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[j][:, h, w], x[j][:, h_prime, w_prime] = x[j][:, h_prime, w_prime], x[j][:, h, w]
        x[j] = torch.from_numpy(np.clip(gaussian(x[j], sigma=per_size[0], channel_axis=0), 0, 1))
    return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = [VGG('VGG16'), DenseNet121(), ResNet101(), MobileNetV2(), VGG('VGG16'), DenseNet121(), ResNet101(), MobileNetV2()]
net_num = len(net)
for i in range(net_num):
    net[i] = net[i].to(device)
    if device == 'cuda':
        net[i] = torch.nn.DataParallel(net[i])
        cudnn.benchmark = True
weight = torch.load('net_weight/' + 'vgg16' + '_ckpt.pth')
net[0].load_state_dict(weight['net'])
net[0].eval()
weight = torch.load('net_weight/' + 'densenet121' + '_ckpt.pth')
net[1].load_state_dict(weight['net'])
net[1].eval()
weight = torch.load('net_weight/' + 'resnet101' + '_ckpt.pth')
net[2].load_state_dict(weight['net'])
net[2].eval()
weight = torch.load('net_weight/' + 'mobilenetv2' + '_ckpt.pth')
net[3].load_state_dict(weight['net'])
net[3].eval()
weight = torch.load('net_weight/gaussnoise_' + 'vgg16' + '_ckpt.pth')
net[4].load_state_dict(weight)
net[4].eval()
weight = torch.load('net_weight/gaussnoise_' + 'densenet121' + '_ckpt.pth')
net[5].load_state_dict(weight)
net[5].eval()
weight = torch.load('net_weight/gaussnoise_' + 'resnet101' + '_ckpt.pth')
net[6].load_state_dict(weight)
net[6].eval()
weight = torch.load('net_weight/gaussnoise_' + 'mobilenetv2' + '_ckpt.pth')
net[7].load_state_dict(weight)
net[7].eval()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
dataiter = iter(testloader)

round_num = 25
img_num = 10
result_density = [[[0 for i in range(img_num)] for j in range(net_num)] for k in range(round_num)]
for jj in range(round_num):
    for ii in range(img_num):
        while True:
            images, labels = dataiter.next()
            with torch.no_grad():
                images, labels = images.to(device), labels.to(device)
                images_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(images)
                all_correct = True
                for i in range(net_num):
                    if net[i](images_norm).max(1)[1][0] != labels:
                        all_correct = False
                        break
                if all_correct:
                    break

        for i in range(net_num):
            per_par = 0.1
            transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            per_img = transform(gauss_noise(images, per_par, 1))
            outputs = net[i](per_img).max(1)[1]
            if outputs == labels:
                result_density[jj][i][ii] = 1
            else:
                result_density[jj][i][ii] = 0

print(result_density)
