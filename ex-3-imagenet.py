from PIL import ImageFile
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
from scipy.special import betaincinv
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
    batch_size = 50

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
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

result_density = [0]*net_num

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
    result_density[i] = robust_density_cal(net[i], images, impulse_noise, per_par, 0.05, 0.05)

print(result_density)






