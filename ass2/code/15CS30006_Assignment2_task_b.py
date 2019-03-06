import numpy as np
import mxnet as mx
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from mxnet import nd, autograd, gluon
import pickle

from utils import DataLoader, weights_download
from networks import network_1, network_2
from modules import My_NAG, My_SGD

weights_download()

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

NUM_EPOCHS = 50

if not os.path.exists("losses/"):
    os.mkdir("losses")

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

def train(epochs, train_data, train_labels, val_data, val_labels, net, params_file,trainer,batch_size=7000):

    loss_train = []
    loss_val = []
    train_data = mx.nd.array(train_data)
    train_labels = mx.nd.array(train_labels)
    val_data = mx.nd.array(val_images)
    val_labels = mx.nd.array(val_labels)
    last_acc = -100
    for e in range(epochs):
        for bt in range(int(len(train_labels)/batch_size)):
            with autograd.record():
                output = net(train_data[bt*batch_size:(bt+1)*batch_size])
                loss = softmax_cross_entropy(output,train_labels[bt*batch_size : (bt+1)*batch_size])
            loss.backward()
            trainer.step(train_data.shape[0])

        acc = mx.metric.Accuracy()
        out_all = net(train_data)
        loss_all = softmax_cross_entropy(out_all,train_labels)
        loss_train.append(loss_all)
        out_val = net(val_data)
        acc.update(preds=nd.argmax(out_val,axis=1),labels=val_labels)
        loss_val.append(softmax_cross_entropy(out_val,val_labels))
        print("Epoch %d : Loss : %f, Val Accuracy : %f"%(e,nd.mean(loss_all).asscalar(),acc.get()[1]))
        if e > 10 and  abs(last_acc - acc.get()[1]) < 0.00001:
            print("Change in Val accuracy < 0.00001, Training FInished!")
            break
        last_acc = acc.get()[1]

    if not os.path.exists("weights/"):
        os.mkdir("weights/")
    print("Saving Model....")
    net.save_parameters(os.path.join("weights",params_file+".params"))

    loss_train = [nd.mean(l).asscalar() for l in loss_train]
    loss_val = [nd.mean(l).asscalar() for l in loss_val]

    return loss_train, loss_val

def test(test_data, test_labels, net, params_file):
    test_data = mx.nd.array(test_data)
    test_labels = mx.nd.array(test_labels)
    net.load_parameters(os.path.join("weights",params_file+".params"))
    output = net(test_data)
    acc = mx.metric.Accuracy()
    acc.update(preds=nd.argmax(output,axis=1),labels=test_labels)
    print("Test Accuracy : %f"%acc.get()[1])

def plot_loss(loss_1, loss_2, filename, label):
    if not os.path.exists("plots/"):
        os.mkdir("plots")

    plt.figure(figsize=(20,15))
    num_eps = len(loss_1)
    plt.plot(list(range(1,num_eps+1)),loss_1)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.savefig("plots/"+filename+"_network1.png")

    plt.figure(figsize=(20,15))
    num_eps = len(loss_2)
    plt.plot(list(range(1,num_eps+1)),loss_2)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.savefig("plots/"+filename+"_network2.png")

if mode == 'train':
    #Experiment 1
    print("Training")
    print("Experiment 1 : Start")
    learning_rate = 0.0001
    print("Normal Initialisation : Start")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp1_normal",trainer)
    with open("losses/exp1_normal_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp1_normal_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val
    print("Normal Initialisation : End")

    print("Xavier Initialisation : Start")
    net = network_2()
    net.collect_params().initialize(mx.init.Xavier())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp1_xavier",trainer)
    with open("losses/exp1_xavier_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp1_xavier_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val
    print("Xavier Initialisation : End")

    print("Orthogonal Initialisation : Start")
    net = network_2()
    net.collect_params().initialize(mx.init.Orthogonal())
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp1_orthogonal",trainer)
    with open("losses/exp1_orthogonal_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp1_orthogonal_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val
    print("Orthogonal Initialisation : End")

    print("Experiment 1 : End")

    print("Experiment 2 : Start")
    print("Using Batch Normalisation")
    net = network_2(batch_norm=True)
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp2",trainer)
    with open("losses/exp2_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp2_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val
    print("Experiment 2 : End")

    print("Experiment 3 : Start")

    print("Dropout : 0.1")
    net = network_2(dropout=0.1)
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp3_dropout10",trainer)
    with open("losses/exp3_dropout10_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp3_dropout10_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Dropout : 0.4")
    net = network_2(dropout=0.4)
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp3_dropout40",trainer)
    with open("losses/exp3_dropout40_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp3_dropout40_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Dropout : 0.6")
    net = network_2(dropout=0.6)
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp3_dropout60",trainer)
    with open("losses/exp3_dropout60_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp3_dropout60_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Experiment 3 : End")

    print("Experiment 4 : Start")

    print("Optimization : SGD")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), My_SGD(learning_rate=learning_rate))
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp4_sgd",trainer)
    with open("losses/exp4_sgd_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp4_sgd_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Optimization : NAG")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), My_NAG(0.9,learning_rate=learning_rate) )
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp4_nag",trainer)
    with open("losses/exp4_nag_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp4_nag_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Optimization : AdaDelta")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), mx.optimizer.AdaDelta(learning_rate=learning_rate))
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp4_adadelta",trainer)
    with open("losses/exp4_adadelta_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp4_adadelta_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Optimization : AdaGrad")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), mx.optimizer.AdaGrad(learning_rate=learning_rate))
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp4_adagrad",trainer)
    with open("losses/exp4_adagrad_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp4_adagrad_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Optimization : RMSProp")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), mx.optimizer.RMSProp(learning_rate))
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp4_rmsprop",trainer)
    with open("losses/exp4_rmsprop_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp4_rmsprop_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Optimization : Adam")
    net = network_2()
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), mx.optimizer.Adam(learning_rate))
    loss_train, loss_val = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2_exp4_adam",trainer)
    with open("losses/exp4_adam_train.list",'wb') as f:
        pickle.dump(loss_train,f)
    with open("losses/exp4_adam_val.list",'wb') as f:
        pickle.dump(loss_val,f)
    del loss_train
    del loss_val

    print("Experiment 4 : End")


if mode == 'test':
    print("Testing")
    print("Experiment 1 : Start")

    print("Normal Initialisation : Start")
    net = network_2()
    test(test_images, test_labels, net, "network2_exp1_normal")
    print("Normal Initialisation : End")

    print("Xavier Initialisation : Start")
    net = network_2()
    test(test_images, test_labels, net, "network2_exp1_xavier")
    print("Xavier Initialisation : End")

    print("Orthogonal Initialisation : Start")
    net = network_2()
    test(test_images, test_labels, net, "network2_exp1_orthogonal")
    print("Orthogonal Initialisation : End")

    print("Experiment 1 : End")

    print("Experiment 2 : Start")
    print("Using Batch Normalisation")
    net = network_2(batch_norm=True)
    test(test_images, test_labels, net, "network2_exp2")
    print("Experiment 2 : End")

    print("Experiment 3 : Start")

    print("Dropout : 0.1")
    net = network_2(dropout=0.1)
    test(test_images, test_labels, net, "network2_exp3_dropout10")

    print("Dropout : 0.4")
    net = network_2(dropout=0.4)
    test(test_images, test_labels, net, "network2_exp3_dropout40")

    print("Dropout : 0.6")
    net = network_2(dropout=0.6)
    test(test_images, test_labels, net, "network2_exp3_dropout60")

    print("Experiment 3 : End")

    print("Experiment 4 : Start")

    print("Optimization : SGD")
    net = network_2()
    test(test_images,test_labels, net, "network2_exp4_sgd")

    print("Optimization : NAG")
    net = network_2()
    test(test_images,test_labels, net, "network2_exp4_nag")

    print("Optimization : AdaDelta")
    net = network_2()
    test(test_images,test_labels, net, "network2_exp4_adadelta")

    print("Optimization : AdaGrad")
    net = network_2()
    test(test_images,test_labels, net, "network2_exp4_adagrad")

    print("Optimization : RMSProp")
    net = network_2()
    test(test_images,test_labels, net, "network2_exp4_rmsprop")

    print("Optimization : Adam")
    net = network_2()
    test(test_images,test_labels, net, "network2_exp4_adam")

    print("Experiment 4 : End")
