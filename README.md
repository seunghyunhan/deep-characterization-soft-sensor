# deep-characterization-soft-sensor

This is code for the paper "Use of Deep Learning for Characterization of Microfluidic Soft Sensors".

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
You can retrieve checkpoints by setting --test flags True.
```bash
python main.py --epoch 100 --test True
```
