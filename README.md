# RXSSNet
The code of paper 'A Novel RGB-X Semantic Segmentation Network With Cross-Modal Feature Reweighting and Local-Global Feature Aggregation'. 

# Abstract
Existing multimodal semantic segmentation methods show significant progress in fusing information from RGB and other specific image types (such as Thermal images). However, these methods often exhibit limited generalization performance when applied to more general RGB-X semantic segmentation tasks. To address this challenge, we propose a novel RGB-X semantic segmentation network that can be generalized well to different cross-modal sensor combinations such as RGB-Thermal, RGB-Depth or RGB-Polarization. In order to ensure that the proposed algorithm can adapt to the fusion of different cross-modal data types, a Cross-Modal Feature Reweighting Module is proposed to adaptively reassign weights to cross-modal features by calculating the cosine similarity of RGB and X feature information. Next, the weighted RGB and X features are fed into the Local-Global Feature Aggregation Module (LAM) for local interaction and fusion. To understand the contextual information more comprehensively, we add Cross-modal Global Prior Information to LAM as a supplement to achieve more robust information fusion. Finally, this paper proposes a Progressive Linear Projection Decoder to improve the segmentation performance through stepwise decoding. Extensive experiments on six multimodal semantic segmentation datasets demonstrate the proposed algorithm's effectiveness and generalization ability, with state-of-the-art results on the MFNet, PST900, FMB, Cityscapes and ZJU datasets. The source code of the algorithm will be made available at https://github.com/poisonzzw/RXSSNet for public access.

# Dataset
The MFNet datesets for RGB-T semantic segmentation could be found in [here](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/).
The PST900 datesets for RGB-T semantic segmentation could be found in [here](https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view?pli=1).
The FMB datesets for RGB-T semantic segmentation could be found in [here](https://pan.baidu.com/s/1k7PgCsSJVZJIoIhgMjWxNg?pwd=IVIF#list/path=%2F).
The Cityscapes datesets for RGB-D semantic segmentation could be found in [here](https://www.cityscapes-dataset.com/dataset-overview/).
The NYUV2 datesets for RGB-D semantic segmentation could be found in [here](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).
The ZJU datesets for RGB-P semantic segmentation could be found in [here](https://huggingface.co/datasets/Zhonghua/ZJU_RGB_P/tree/main).

# Implementation details
1. We have tested the following versions of OS and softwares:

    • OS: Windows 11

    • CUDA: 112.6  

    • PyTorch 1.12.1  

    • Python 3.9  

2. Install all dependencies. Install pytorch, cuda and cudnn, then install other dependencies via:

   pip install -r requirements.txt

# Pretrained model weight
Download the pretrained ConvNext V2 - tiny here. [pretrained ConvNext V2](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt).

# Result weight
RXSSNet's model weights on six datasets can be downloaded here. [Result weight](https://pan.baidu.com/s/1_1m1YiBUoR3nXqSGU_st_A). 提取码: s8q8


# Evaluation
Download the corresponding dataset and weight file, modify the corresponding path, and run the test.py.

# Contact
Please drop me an email for further problems or discussion: zhangzhiwei@mail.dlut.edu.cn
