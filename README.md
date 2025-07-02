# RXSSNet
# MS-IRTNet
The code of paper 'MS-IRTNet: Multi-Stage Information Interaction Network for RGB-T Semantic Segmentation'. 

# Abstract
The complementary information from RGB and thermal images can remarkably boost semantic
segmentation performance. Existing RGB-T segmentation methods usually use simple interaction
strategies to extract complementary information from RGB and thermal images, which ignores
recognizability features from different imaging mechanisms. To address these problems, we
propose a multistage information interaction network for RGB-T semantic segmentation called
MS-IRTNet. MS-IRTNet has a dual-stream encoder structure that can extract multistage feature
information. To better interact with multimodal information, we design a gate-weighted
interaction module (GWIM) and a feature information interaction module (FIIM). GWIM can
learn multimodal information weights in different channels, while FIIM integrates and fuses
weighted RGB and thermal information into a single feature map. Finally, multistage interactive
information is fed into the decoder for semantic prediction. Our method achieves 60.5 mIoU on
the MFNet dataset, outperforming state-of-the-art methods. Notably, MS-IRTNet also achieved
state-of-the-art results in tests of daytime images (51.7 mIoU) and nighttime images (62.5 mIoU).
The code and pre-trained models are available at https://github.com/poisonzzw/MS-IRTNet.

# Requirements
CUDA 11.2，torchvision 0.13.1，Tensorboard 2.9.0，Python 3.9，PyTorch 1.12.1。

# Dataset
The MFNet datesets for RGB-T semantic segmentation could be found in [here](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)。
The Data.Dataloader_voxel file consists of two parts: data import and data augmentation, which can be written by the reader. Since the research group currently has a paper under review, if it is accepted, we will upload the file immediately.

# Result
Predict maps: [百度网盘](https://pan.baidu.com/s/1T4J-iTgW7nBZWcCTmNsIBQ).
提取码rd60。

# train
Download the pretrained ConvNext V2 - tiny here [pretrained ConvNext V2](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt).



# test 
在Train.py文件中注释train代码部分，运行test部分，导入保存的整体模型即可。
Model weights download：[百度网盘](https://pan.baidu.com/s/1jhBzhxnfD2_oOhnnTF0zCQ).
提取码zk9s。



# Citation
@article{zhang2023ms,
  title={MS-IRTNet: Multistage information interaction network for RGB-T semantic segmentation},
  author={Zhang, Zhiwei and Liu, Yisha and Xue, Weimin},
  journal={Information Sciences},
  volume={647},
  pages={119442},
  year={2023},
  publisher={Elsevier}
}

# Contact
Please drop me an email for further problems or discussion: 1519968317@qq.com
