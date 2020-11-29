# 魅力-语义分割模型

  
本库包含有常见的语义分割网络模型，例如FCN，U-Net，DeepLab系列，ResNet系列，ENet，MobileNet系列，PASNet，PSPNet，HRNet，ICNet，AttUNet，NesterUNet，SEUNet，VGG，R2UNet，SegNet等网络。包含有多种损失函数选择，多种学习率调整方式。  

## data
**build:**  

------train:  
------------images  
------------labels    
------val:  
---------images  
---------labels    
------test:  
----------images  
----------labels    
**ccd:**  
------train:  
------------images  
------------labels    
------val:  
---------images  
---------labels    
------test:  
----------images  
----------labels    
**camvid:**  
------train:  
------------images  
------------labels    
------val:  
---------images  
---------labels    
------test:  
----------images  
----------labels   

## loss
包含有各种损失函数，例如

```python
from losses.B_Focal_loss import focal_loss_binary
from losses.C_Focal_loss import focal_loss_multiclasses
from losses.Dice_loss import DiceLoss
from losses.BCE_Dice_loss import BCE_DiceLoss
from losses.CE_Dice_loss import CE_DiceLoss
from losses.Tversky_loss import TverskyLoss
from losses.Focal_Tversky_loss import FocalTverskyLoss
from losses.Weighted_Categorical_loss import Weighted_Categorical_CrossEntropy_Loss
from losses.Generalized_Dice_loss import GeneralizedDiceLoss
from losses.Jaccard_loss import JaccardLoss
from losses.BCE_Jaccard_Loss import BCE_JaccardLoss
from losses.CE_Jaccard_Loss import CE_JaccardLoss
```

## metrics

```python
'iou_score', 'jaccard_score', 'f1_score', 'f2_score',
'dice_score', 'get_f_score', 'get_iou_score', 'get_jaccard_score'
```
## lr

```python
CosineDecayScheduler
CosineDecayScheduler+
ReduceLROnPlateau
lr
```
