from __future__ import absolute_import

from .ENet import *
from .FCN8 import *
from .VGGFCN8 import *
from .ICNet import *
from .MobileNetFCN8 import *
from .MobileNetUnet import *
from .PSPNet import *
from .Segnet import *
from .SEUNet import *
from .UNet import *
from .R2UNet import *
from .R2AttUNet import *
from .NestedUNet import *
from .AttUNet import *
from .scSEUnet import *
from .VGGUnet import *
from .DeepLabV2 import *
from .UNet_Xception_ResNetBlock import *
from .HRNet import *
from .DeepLab_mobilenet import *
from .DeepLab_xception import *
from .DeepLabV3plus import *
from .UNet_mobilenet import *
from .SegNet_resnet import resnet50_segnet
from .SegNet_mobilenet import mobilenet_segnet
from .PSPNet_mobilenet import mobilenet_pspnet


__model_factory = {
    'enet': ENet,
    'fcn8': FCN8,
    'mobilenet_fcn8': MobileNetFCN8,
    'vggfcn8': VGGFCN8,
    'unet': UNet,
    'attunet': AttUNet,
    'r2unet': R2UNet,
    'r2attunet': R2AttUNet,
    'vggunet': VGGUnet,
    'unet_xception_resnetblock': Unet_Xception_ResNetBlock,
    'mobilenet_unet': MobileNetUnet,
    'seunet': SEUnet,
    'scseunet': scSEUnet,
    'segnet': Segnet,
    'pspnet': PSPNet,
    'icnet': ICNet,
    'deeplab_v2': DeepLabV2,
    'hrnet': HRNet,
    'unet++': NestedUNet,
    'DeepLab_mobilenet': DeepLabv3plus_mobilenetV2
}
