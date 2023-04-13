import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from scipy.special import betaincinv
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2
from skimage.filters import gaussian


def okamoto(eps, delta):
    return math.ceil(math.log(2 / delta) / (2 * eps * eps))


def strong(p, eps):
    # f(p,eps)
    assert eps < 1 - p
    return (p / (p + eps)) ** (p + eps) * ((1 - p) / (1 - p - eps)) ** (1 - p - eps)


def binary(f1, f2, delta, low, high):
    # find the least n, s.t. f1^n+f2^n<=delta
    assert f1**high + f2**high <= delta
    while high - low > 1:
        mid = (high + low) // 2
        if f1 ** mid + f2 ** mid > delta:
            low = mid
        else:
            high = mid
    return high


def bound_estimate(cp_l, cp_h, delta, eps):
    if cp_h <= eps:
        f = strong(cp_h, eps)
        return math.ceil(math.log(delta, f))
    elif cp_l >= 1 - eps:
        f = strong(1 - cp_l, eps)
        return math.ceil(math.log(delta, f))
    elif (1 - eps) / 2 <= cp_h <= (1 + eps) / 2 or (1 - eps) / 2 <= cp_l <= (1 + eps) / 2 or (
            cp_h >= (1 + eps) / 2 and cp_l <= (1 - eps) / 2):
        return okamoto(eps, delta)
    else:
        if eps < cp_h < (1 - eps) / 2:
            pp = cp_h
        else:
            assert (1 + eps) / 2 < cp_l < 1 - eps
            pp = cp_l
        f1 = strong(pp, eps)
        f2 = strong(1 - pp, eps)
        return binary(f1, f2, delta, 0, okamoto(eps, delta))


def cp_int(N, Np, delta):
    # Clopper-Pearson confidence interval
    if Np == 0:
        cp_l = 0
    else:
        cp_l = betaincinv(Np, N - Np + 1, delta / 2)
    if Np == N:
        cp_h = 1
    else:
        cp_h = betaincinv(Np + 1, N - Np, 1 - delta / 2)
    return cp_l, cp_h


def robust_density_cal(net, x, per_gen, per_par, delta, eps):
    delta1 = 0.05 * delta
    assert 0 < eps < 1 / 3

    M = okamoto(eps, delta)
    batch_size = 100

    # first stage
    size_1 = max(min(math.ceil(0.01*M), 100), 10)
    result = DNN_sample(net, x, per_gen, per_par, batch_size, size_1)
    p_1 = result / size_1

    # second stage
    N_list = [round((i + 1) * M / 100) for i in range(20)]  # 20 candidates
    length = len(N_list)
    N_num = N_list.copy()  # total_cost
    for i in range(length):
        N = N_list[i]
        Np = round(N * p_1)
        cp_l, cp_h = cp_int(N, Np, delta1)
        N_num[i] += bound_estimate(cp_l, cp_h, (delta - delta1) / (1 - delta1), eps)

    m = min(N_num)

    # 第二轮采样
    if m < M:
        size_2 = N_list[N_num.index(m)]
        cp_l, cp_h = cp_int(size_2, DNN_sample(net, x, per_gen, per_par, batch_size, size_2), delta1)
        size_3 = bound_estimate(cp_l, cp_h, (delta - delta1) / (1 - delta1), eps)
        return 1 - DNN_sample(net, x, per_gen, per_par, batch_size, size_3) / size_3
    else:
        return 1 - DNN_sample(net, x, per_gen, per_par, batch_size, M) / M


def DNN_sample(_net, x, per_gen, per_size, batch_size, total):
    x_raw = x
    count = 0
    for i in range((total - 1) // batch_size + 1):
        if i == (total - 1) // batch_size:
            num = total - batch_size * ((total - 1) // batch_size)
        else:
            num = batch_size
        per_x = per_gen(x_raw, per_size, num)
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        x = transform(x_raw)
        per_x = transform(per_x)
        _net.eval()
        with torch.no_grad():
            x, per_x = x.to(device), per_x.to(device)
            outputs_x = _net(x)
            _, predicted_x = outputs_x.max(1)
            outputs = _net(per_x)
            _, predicted = outputs.max(1)
            result = (predicted == predicted_x)
            count += result.count_nonzero().item()
    return total - count


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
weight = torch.load('net_weight/glassblur_' + 'vgg16' + '_ckpt.pth')
net[4].load_state_dict(weight)
net[4].eval()
weight = torch.load('net_weight/glassblur_' + 'densenet121' + '_ckpt.pth')
net[5].load_state_dict(weight)
net[5].eval()
weight = torch.load('net_weight/glassblur_' + 'resnet101' + '_ckpt.pth')
net[6].load_state_dict(weight)
net[6].eval()
weight = torch.load('net_weight/glassblur_' + 'mobilenetv2' + '_ckpt.pth')
net[7].load_state_dict(weight)
net[7].eval()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
dataiter = iter(testloader)

img_num = 10
result_density = [[0 for i in range(img_num)] for j in range(net_num)]
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
        per_par = (0.4,1,2)
        result_density[i][ii] = robust_density_cal(net[i], images, glass_blur, per_par, 0.05, 0.05)

print(result_density)
