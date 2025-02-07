"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import sys
import os

import input_data

from foolbox import PyTorchModel, Model
from foolbox.attacks import LinfPGD, FGSM, FGM, L2PGD, L1PGD, L1FastGradientAttack


def test(config):

    # Setting up testing parameters
    seed = config['random_seed']
    batch_size = config['training_batch_size']
    backbone_name = config['backbone']

    if not os.path.isfile(config["model_dir"] + '/results/training.done'):
        print("Not trained") 
        return

    if os.path.isfile(config["model_dir"] + '/results/testing.done') and not config["restart"]:
        print("Already tested")
        return

    # Setting up the data and the model
    data = input_data.load_data_set(results_dir=config['results_dir'], data_set=config['data_set_id'],
                                    standarized=config["standarize"], multiplier=config["standarize_multiplier"],
                                    re_size=config["re_size"], seed=seed)

    num_features = data.train.images.shape[1]

    if config['data_set_id']==67: #cifar
        s = 32*32*3
    else:
        s = 28*28

    if config['backbone'] == 'ThreeLayer':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(s, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        ).cuda()

    model.load_state_dict(torch.load(config['model_dir'] + '/checkpoints/checkpoint.pth'))
    #model.eval()
    #Setting up attacks

    pre = dict(std=None, mean=None)  # RGB to BGR
    fmodel: Model = PyTorchModel(model, bounds=(config["bound_lower"], config["bound_upper"]), preprocessing=pre)
    fmodel = fmodel.transform_bounds((config["bound_lower"], config["bound_upper"]))

    #fmodel = foolbox.PyTorchModel(model, bounds=(config["bound_lower"], config["bound_upper"]))

    epsilons_inf = [
            0.0,
            0.0002,
            0.0005,
            0.0008,
            0.001,
            0.0015,
            0.002,
            0.003,
            0.01,
            0.1,
            0.3,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            50.0
        ]


    epsilons_l2 = list(np.sqrt(num_features) * np.array(epsilons_inf))

    attacks = [L1PGD(), L1FastGradientAttack(), LinfPGD(), L2PGD(), FGSM(), FGM()]
    name_attacks = ["l1_pgd_norm", "l1_fgm_norm", "linf_pgd", "l2_pgd_norm", "linf_fgsm", "l2_fgm_norm"]
    epsilons = [epsilons_l2, epsilons_l2, epsilons_inf, epsilons_l2, epsilons_inf, epsilons_l2]

    num_iter = int(10)# * (int(256/batch_size)))

    for attack, name_attack, epsilon in zip(attacks, name_attacks, epsilons):

        for dataset in ["val", "test"]:

            for iter in range(num_iter):

                if dataset == "val":
                    x_batch, y_batch = data.validation.next_batch(batch_size)
                else:
                    x_batch, y_batch = data.test.next_batch(batch_size)

                _, _, success = attack(fmodel, torch.from_numpy( x_batch ).float().cuda() , torch.from_numpy(y_batch).cuda(),
                                                         epsilons=epsilon)

                robust_accuracy = 1 - success.cpu().numpy().mean(axis=-1)

                if iter == 0:
                    acc = dict(zip(epsilon, robust_accuracy))
                else:
                    tmp = dict(zip(epsilon, robust_accuracy))
                    acc = {k: acc[k] + tmp[k] for k in acc.keys()}

            acc = {k: acc[k]/num_iter for k in acc.keys()}
            print(acc)
            with open(config['model_dir'] + '/results/acc_' + dataset + '_' + name_attack + '.pkl', 'wb') as f:
                pickle.dump(acc, f)

        print("\n Attack " + name_attack + " done")
        sys.stdout.flush()

    open(config['model_dir'] + '/results/testing.done', 'w').close()




