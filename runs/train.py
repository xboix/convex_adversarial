import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle

import problems as pblm
from trainer import *
import math
import numpy as np
import os


class cArgs:
    def __init__(self, batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-3,
                 epsilon=0.1, starting_epsilon=None,
                 proj=None,
                 norm_train='l1', norm_test='l1',
                 opt='sgd', momentum=0.9, weight_decay=5e-4):

        self.opt = opt
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.test_batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        # epsilon settings
        self.epsilon = epsilon
        self.starting_epsilon = starting_epsilon
        self.schedule_length = 10

        # projection settings
        self.proj = proj
        self.norm_train = norm_train
        self.norm_test = norm_test

        # model arguments
        self.model = None
        self.model_factor = 8
        self.cascade = 1
        self.method = None
        self.resnet_N = 1
        self.resnet_factor = 1

        # other arguments
        self.seed = seed
        self.real_time = True
        self.cuda_ids = None
        self.prefix = None
        self.load = None
        self.verbose = verbose

        if self.starting_epsilon is None:
            self.starting_epsilon = self.epsilon
        if self.prefix:
            if self.model is not None:
                self.prefix += '_' + self.model

            if self.method is not None:
                self.prefix += '_' + self.method

            banned = ['verbose', 'prefix',
                      'resume', 'baseline', 'eval',
                      'method', 'model', 'cuda_ids', 'load', 'real_time',
                      'test_batch_size']
            if self.method == 'baseline':
                banned += ['epsilon', 'starting_epsilon', 'schedule_length',
                           'l1_test', 'l1_train', 'm', 'l1_proj']

            # Ignore these parameters for filename since we never change them
            banned += ['momentum', 'weight_decay']

            if self.cascade == 1:
                banned += ['cascade']

            # if not using a model that uses model_factor,
            # ignore model_factor
            if self.model not in ['wide', 'deep']:
                banned += ['model_factor']

            # if args.model != 'resnet':
            banned += ['resnet_N', 'resnet_factor']

            for arg in sorted(vars(self)):
                if arg not in banned and getattr(self, arg) is not None:
                    self.prefix += '_' + arg + '_' + str(getattr(self, arg))

            if self.schedule_length > self.epochs:
                raise ValueError('Schedule length for epsilon ({}) is greater than '
                                 'number of epochs ({})'.format(self.schedule_length, self.epochs))
        else:
            self.prefix = 'temporary'

        if self.cuda_ids is not None:
            print('Setting CUDA_VISIBLE_DEVICES to {}'.format(self.cuda_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_ids


def train_baseline(loader, model, opt, epoch, log, verbose, standarization):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):

        if standarization:
            m = np.mean(X.numpy(), axis=(1, 2, 3))
            X = X - m[:, np.newaxis, np.newaxis, np.newaxis]
            d = np.std(X.numpy(), axis=(1, 2, 3))
            X = X / d[:, np.newaxis, np.newaxis, np.newaxis]

        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)

        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))

        print(epoch, i, ce.data, err, file=log)
        if verbose and i % verbose == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors))
        log.flush()


def train(config):
    # Setting up training parameters
    args = cArgs(batch_size=config['training_batch_size'], epochs=config['max_num_training_steps'],
                 opt='adam', verbose=200, starting_epsilon=0.01, epsilon=config['epsilon'], lr=config['initial_learning_rate'])
    print("saving file to {}".format(config["model_dir"]))
    setproctitle.setproctitle(config["model_dir"])
    train_log = open(config["model_dir"] + "/train.log", "w")
    test_log = open(config["model_dir"] + "/test.log", "w")


    if config["skip"]:
        print("SKIP")
        return

    backbone_name = config['backbone']
    robust_training = config['robust_training']

    if os.path.isfile(config["model_dir"] + '/results/training.done') and not config["restart"]:
        print("Already trained")
        return

    # Setting up the data and the model
    if config['data_set'] == 'mnist':

        train_loader, _ = pblm.mnist_loaders(args.batch_size)
        _, test_loader = pblm.mnist_loaders(args.test_batch_size)

    elif config['data_set'] == 'fashion':

        def fashion_loaders(batch_size, shuffle_test=False):
            mnist_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
            mnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                                       pin_memory=True)
            test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test,
                                                      pin_memory=True)
            return train_loader, test_loader

        train_loader, _ = fashion_loaders(args.batch_size)
        _, test_loader = fashion_loaders(args.test_batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for X, y in train_loader:
        break
    kwargs = pblm.args2kwargs(args, X=Variable(X.cuda()))
    best_err = 1


    model_dir = config['model_dir']
    start_iteration = 0

    if config['backbone'] == 'ThreeLayer':
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    sampler_indices = []
    model = [select_model(args.model)]

    # Main training loop
    if args.opt == 'adam':
        opt = optim.Adam(model[-1].parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        opt = optim.SGD(model[-1].parameters(), lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    eps_schedule = np.linspace(args.starting_epsilon,
                               args.epsilon,
                               args.schedule_length)

    for t in range(args.epochs):
        lr_scheduler.step(epoch=max(t - len(eps_schedule), 0))
        if t < len(eps_schedule) and args.starting_epsilon is not None:
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        # standard training
        if config["robust_training"]:
            train_robust(train_loader, model[0], opt, epsilon, t,
                         train_log, args.verbose, args.real_time,
                         norm_type=args.norm_train, bounded_input=True, **kwargs)
        else:
            train_baseline(train_loader, model[0], opt, t, train_log, args.verbose)

        torch.save({
            'state_dict': model.state_dict(),
            'epoch': t
        }, config["model_dir"] + "_checkpoint.pth")

    # Flag the training completed and store the training time profile
    open(model_dir + '/results/training.done', 'w').close()


