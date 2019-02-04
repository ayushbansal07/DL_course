import mxnet as mx
from mxnet import nd, autograd, gluon

#Network 1
def network_1():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(64, activation="relu"))
        net.add(gluon.nn.Dense(32, activation="relu"))
        net.add(gluon.nn.Dense(16, activation="relu"))
        net.add(gluon.nn.Dense(10))

    return net

#Network 2
def network_2(batch_norm=False,dropout = 0):
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1024, activation="relu"))
        if batch_norm:
            net.add(gluon.nn.BatchNorm())
        if dropout != 0:
            net.add(gluon.nn.Dropout(dropout))
        net.add(gluon.nn.Dense(512, activation="relu"))
        if batch_norm:
            net.add(gluon.nn.BatchNorm())
        if dropout != 0:
            net.add(gluon.nn.Dropout(dropout))
        net.add(gluon.nn.Dense(256, activation="relu"))
        if batch_norm:
            net.add(gluon.nn.BatchNorm())
        if dropout != 0:
            net.add(gluon.nn.Dropout(dropout))
        net.add(gluon.nn.Dense(10))

    return net
