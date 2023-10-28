# Effective cross-sensor color constancy using a dual-mapping strategy(Pre-release version)
Shuwei Yue and *[Minchen Wei](https://www.polyucolorlab.com/)

*Color, Imaging, and Metaverse Research Center, The Hong Kong Polytechnic University.*

![DMCC-Overview](https://github.com/shuwei666/DMCC-Cross-sensor-color-constancy/assets/106613332/105912f5-b9a2-41ba-8477-54ddefa48cd0)

## Code
#### Prerequisite
- Pytorch
- opencv-python

#### Training
Choose the training sensor and testing sensor in `get_args()`function of `train.py`

run `python train.py`


#### Testing
Define the testing sensor and selecting the related pretraind model

run `python test.py`

