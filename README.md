# Effective cross-sensor color constancy using a dual-mapping strategy
[Shuwei Yue](https://shuwei666.github.io/) and *[Minchen Wei](https://www.polyucolorlab.com/)

*Color, Imaging, and Metaverse Research Center, The Hong Kong Polytechnic University.*

![DMCC-Overview](https://github.com/shuwei666/DMCC-Cross-sensor-color-constancy/assets/106613332/105912f5-b9a2-41ba-8477-54ddefa48cd0)
## BibTex
Please cite us if you use this code or our paper:
>@article{Yue24,  
author = {Shuwei Yue and Minchen Wei},  
journal = {J. Opt. Soc. Am. A},
number = {2},  
pages = {329--337},  
publisher = {Optica Publishing Group},  
title = {Effective cross-sensor color constancy using a dual-mapping strategy},  
volume = {41},  
month = {Feb},  
year = {2024}  
}


## PDF
[PDF(Manuscript )](https://connectpolyu-my.sharepoint.com/personal/21064184r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F21064184r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FPapers%20and%20PPTs%2FYue%20and%20Wei%20%2D%202023%20%2D%20Effective%20cross%2Dsensor%20color%20constancy%20using%20a%20dua%2Epdf&parent=%2Fpersonal%2F21064184r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FPapers%20and%20PPTs&ga=1)

## Code
#### Prerequisite
- Pytorch
- opencv-python

#### Training
Choose the training sensor and testing sensor in `get_args()`function of `train.py`

run `python train.py`


#### Testing
Define the testing sensor and selecting the related pre-trained model

run `python test.py`

#### Frequent questions

1. How to train your own dataset

- step1: generate the diagonal matrix(need a white point under 6500K as described in our paper)
- step2: prepare your own testing data

2. Sample code

    TBD
