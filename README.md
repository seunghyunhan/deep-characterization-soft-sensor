# deep-characterization-soft-sensor

This is code for the paper: <br>
*[Use of Deep Learning for Characterization of Microfluidic Soft Sensors](https://ieeexplore.ieee.org/abstract/document/8255560), **Seunghyun Han**, Taekyoung Kim, Dooyoung Kim, Yong-Lae Park and, Sungho Jo, IEEE Robotics and Automation Letter (RA-L), also selected by ICRA'2018*

We implemented this code, under the following packages:
* Python 3.5.4
* Tensorflow 
* Numpy

## How to train
```bash
python main.py --epoch 100
```

## How to test
After training, it saves checkpoint in the ./ckpt folder.
You can retrieve checkpoints by setting --test flag 'True'.
```bash
python main.py --epoch 100 --test True
```
