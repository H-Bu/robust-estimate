# robustness-estimate
Proofs of Theorems 3-5 can be found in "supplement.pdf"

Implementations for perturbations come from [CIFAR-10-C(ImageNet-C)](https://github.com/hendrycks/robustness). Implementations for networks (VGG16, DenseNet121, ResNet101 and MobileNetV2) for CIFAR-10 come from [here](https://github.com/kuangliu/pytorch-cifar).

Environment:
```
numpy=1.22.3
python==3.10.4
pytorch==1.12.0
scikit-image==0.19.2
torchvision==0.13.0
```

Experiment 2:

Pre-trained networks for ImageNet (from torchvision):
```
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights
net = mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)
net = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
net = regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V1)
```
Experiment 3:

delta=epsilon=0.05

Parameters for perturbations:
```
CIFAR-10:
contrast: uniform distribution in [0,0.5]
fog: (1.5,1.75)
Gaussian noise: 0.1
glass blur: (0.4,1,2)
ImageNet:
contrast: uniform distribution in [0,0.5]
impulse noise: 0.06
shot noise: 20
```

Pre-trained networks for ImageNet (from torchvision):
```
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
net = [shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
       resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1),
       vgg13(weights=VGG13_Weights.IMAGENET1K_V1), vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1),
       vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1), vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
       mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1), vgg19(weights=VGG19_Weights.IMAGENET1K_V1),
       regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V1), resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
       vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1), mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)]
```
