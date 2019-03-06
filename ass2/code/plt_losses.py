import matplotlib
import matplotlib.pyplot as plt
import pickle
import os

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def plot_losses(loss_list, legend_names, filename, label):
    if not os.path.exists("plots/"):
        os.mkdir("plots")

    plt.figure(figsize=(20,15))
    plt.xlabel('Epochs')
    plt.ylabel(label)
    for loss in loss_list:
        num_eps = len(loss)
        plt.plot(list(range(1,num_eps+1)),loss)
    plt.legend(legend_names)
    plt.savefig("plots/"+filename)

############################Experiment 1######################################################
losses = []
with open('losses/exp1_normal_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp1_orthogonal_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp1_xavier_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["Normal Initialisation","Orthogonal Initialisation", "Xavier Initialisation"],"task_b_exp1_train.png","Training Loss")

losses = []
with open('losses/exp1_normal_val.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp1_orthogonal_val.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp1_xavier_val.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["Normal Initialisation","Orthogonal Initialisation", "Xavier Initialisation"],"task_b_exp1_val.png","Validation Loss")


############################Experiment 2######################################################
losses = []
with open('losses/exp2_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp2_val.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["Batch Normalisation Train Loss","Batch Normalisation Validation Loss"],"task_b_exp2.png","Loss")


############################Experiment 3######################################################
losses = []
with open('losses/exp1_normal_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp3_dropout10_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp3_dropout40_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp3_dropout60_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["Dropout 0","Dropout 0.1","Dropout 0.4", "Dropout 0.6"],"task_b_exp3_train.png","Training Loss")

losses = []
with open('losses/exp1_normal_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp3_dropout10_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp3_dropout40_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp3_dropout60_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["Dropout 0","Dropout 0.1","Dropout 0.4", "Dropout 0.6"],"task_b_exp3_val.png","Validation Loss")


############################Experiment 4######################################################
losses = []
with open('losses/exp4_sgd_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_nag_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_adadelta_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_adagrad_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_rmsprop_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_adam_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["SGD","Nestrov's Accelerated Momentum","AdaDelta", "AdaGrad", "RMSProp", "Adam"],"task_b_exp4_train.png","Training Loss")

losses = []
with open('losses/exp4_sgd_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_nag_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_adadelta_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_adagrad_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_rmsprop_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
with open('losses/exp4_adam_train.list','rb') as f:
    l = pickle.load(f)
    losses.append(l)
plot_losses(losses,["SGD","Nestrov's Accelerated Momentum","AdaDelta", "AdaGrad", "RMSProp", "Adam"],"task_b_exp4_val.png","Validation Loss")
