# Value Iteration Networks

This is an experimental code for reproducing below paper's result using chainer. 

```
A. Tamar, Y. Wu, G. Thomas, S. Levine, P. Abbeel, Value Iteration Networks,
Neural Information Processing Systems (NIPS) 2016.
```

## Preparation 

This is preparation code for value iteration networks. 
This script generats grid word and training data of shortest path.
After running this script, map_data.pkl is generated at current directory.

```
python script_make_data.py
```

## Training

We assume that you already have run preparation script and had the map_data.pkl.
This script trains VIN network and generates trained weight in a result directory.

```
python train.py --gpu 0
```

## Test

This code is testing script of VIN.

```
python predict.py --model [trained weight]
```