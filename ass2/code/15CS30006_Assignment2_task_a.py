import numpy as np
import mxnet as mx
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
from mxnet import nd, autograd, gluon

from utils import DataLoader
from networks import network_1, network_2

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

NUM_EPOCHS = 100

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

def train(epochs, train_data, train_labels, val_data, val_labels, net, params_file,learning_rate=0.001,batch_size=7000):
    net.collect_params().initialize(mx.init.Normal(sigma=.1))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
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
        print("Epoch %d : Loss : %f, Val Accuracy : %f"%(e,nd.sum(loss_all).asscalar(),acc.get()[1]))
        if e > 10 and  abs(last_acc - acc.get()[1]) < 0.00001:
            print("Change in Val accuracy < 0.00001, Training FInished!")
            break
        last_acc = acc.get()[1]

    if not os.path.exists("weights/"):
        os.mkdir("weights/")
    print("Saving Model....")
    net.save_parameters(os.path.join("weights",params_file+".params"))

    loss_train = [nd.sum(l).asscalar() for l in loss_train]
    loss_val = [nd.sum(l).asscalar() for l in loss_val]

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
    #plt.legend(["Network 1", "Network 2"])



if mode == 'train':
    net = network_1()
    print("Training Network 1....")
    loss_train1, loss_val1 = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network1")
    print("Finished Training Network 1")
    net = network_2()
    print("Training Network 2....")
    loss_train2, loss_val2 = train(NUM_EPOCHS, train_images, train_labels, val_images, val_labels, net, "network2",0.0001)
    print("Finished Training Network 2")
    print("Plotting Losses")
    plot_loss(loss_train1, loss_train2,"task_a_training_loss", "Training Loss")
    plot_loss(loss_val1, loss_val2, "task_a_validation_loss", "Validation Loss")

if mode == 'test':
    net1 = network_1()
    print("Testing on Network 1....")
    test(test_images, test_labels, net1, "network1")
    net2 = network_2()
    print("Testing on Network 2....")
    test(test_images, test_labels, net2, "network2")
