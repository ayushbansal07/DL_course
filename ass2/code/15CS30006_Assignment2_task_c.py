import numpy as np
import mxnet as mx
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from mxnet import nd, autograd, gluon
from sklearn.linear_model import LogisticRegression
import pickle

from utils import DataLoader, weights_download
from networks import network_1, network_2

weights_download()

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

NUM_EPOCHS = 10

parser = argparse.ArgumentParser()
g = parser.add_mutually_exclusive_group()
g.add_argument("--train",action='store_true')
g.add_argument("--test",action='store_true')
args = parser.parse_args()
mode = 'train'
if args.test:
    mode = 'test'
print("Mode :",mode)

data_loader = DataLoader()
train_images, train_labels = data_loader.load_data('train')
test_images, test_labels = data_loader.load_data('test')

NUM_TRAIN = int(.7*len(train_labels))
NUM_TRAIN

#SHUFFLE
np.random.seed(42)
perm = np.random.permutation(train_images.shape[0])
train_images = train_images[perm]
train_labels = train_labels[perm]

#Split into train and val
val_images = train_images[NUM_TRAIN:]
val_labels = train_labels[NUM_TRAIN:]
train_images = train_images[:NUM_TRAIN]
train_labels = train_labels[:NUM_TRAIN]

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

net = network_2()
net.load_parameters(os.path.join("weights","network2.params"))
params_dict = net.collect_params()
all_params = []
print("List of params found in Network 2 : ")
for key, val in params_dict.items():
    all_params.append(val.data().asnumpy())
    print(key)
print("========================================")

w1 = all_params[0]
b1 = all_params[1]
w2 = all_params[2]
b2 = all_params[3]
w3 = all_params[4]
b3 = all_params[5]

if mode == 'train':

    print("Preparing Layer 1 features...")
    layer_1_feats = train_images.dot(w1.T) + b1
    layer_1_feats = np.maximum(0,layer_1_feats)

    print("Training Layer 1 model : Start")
    model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter = 50,verbose=1)
    model.fit(layer_1_feats,train_labels)
    print("Training Layer 1 model : End")
    with open('weights/taskc_layer1.model','wb') as f:
        pickle.dump(model,f)

    print("Preparing Layer 2 features...")
    layer_2_feats = layer_1_feats.dot(w2.T) + b2
    layer_2_feats = np.maximum(0,layer_2_feats)

    print("Training Layer 2 model : Start")
    model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter = 50,verbose=1)
    model.fit(layer_2_feats,train_labels)
    print("Training Layer 2 model : End")
    with open('weights/taskc_layer2.model','wb') as f:
        pickle.dump(model,f)

    print("Preparing Layer 3 features...")
    layer_3_feats = layer_2_feats.dot(w3.T) + b3
    layer_3_feats = np.maximum(0,layer_3_feats)

    print("Training Layer 3 model : Start")
    model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter = 50,verbose=1)
    model.fit(layer_3_feats,train_labels)
    print("Training Layer 3 model : End")
    with open('weights/taskc_layer3.model','wb') as f:
        pickle.dump(model,f)

if mode == 'test':
    print("Preparing Layer 1 features...")
    layer_1_feats = test_images.dot(w1.T) + b1
    layer_1_feats = np.maximum(0,layer_1_feats)

    with open('weights/taskc_layer1.model','rb') as f:
        model = pickle.load(f)
    predictions = model.predict(layer_1_feats)
    acc = np.sum(predictions==test_labels)/len(test_labels)

    print("Layer 1 model Acuuracy : ",acc)


    print("Preparing Layer 2 features...")
    layer_2_feats = layer_1_feats.dot(w2.T) + b2
    layer_2_feats = np.maximum(0,layer_2_feats)

    with open('weights/taskc_layer2.model','rb') as f:
        model = pickle.load(f)
    predictions = model.predict(layer_2_feats)
    acc = np.sum(predictions==test_labels)/len(test_labels)
    print("Layer 2 model Acuuracy : ",acc)


    print("Preparing Layer 3 features...")
    layer_3_feats = layer_2_feats.dot(w3.T) + b3
    layer_3_feats = np.maximum(0,layer_3_feats)

    with open('weights/taskc_layer3.model','rb') as f:
        model = pickle.load(f)
    predictions = model.predict(layer_3_feats)
    acc = np.sum(predictions==test_labels)/len(test_labels)
    print("Layer 3 model Acuuracy : ",acc)
