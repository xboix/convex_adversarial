import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import sys
import time

from convex_adversarial import robust_loss, robust_loss_parallel

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import math
import numpy as np
import os
from timeit import default_timer as timer
import pickle

class cArgs:
    def __init__(self, batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-3,
                 epsilon=0.1, starting_epsilon=None,
                 proj=None,
                 norm_train='l1', norm_test='l1',
                 opt='adam', momentum=0.9, weight_decay=5e-4):

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mnist_loaders(batch_size, shuffle_test=False):
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def fashion_mnist_loaders(batch_size, shuffle_test=False):
    mnist_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def cifar_loaders(batch_size, shuffle_test=False):
    train = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)

    return train_loader, test_loader


def args2kwargs(args, X=None):

    if args.proj is not None:
        kwargs = {
            'proj' : args.proj,
        }
    else:
        kwargs = {
        }
    kwargs['parallel'] = (args.cuda_ids is not None)
    return kwargs


def train_baseline(loader, model, opt, epoch, log, verbose, standarization, color):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):

        if color:
            X = X.permute(0, 2, 3, 1)

        if standarization==True:
            if not color:
                m = np.mean(X.numpy(), axis=(1, 2, 3))
                X = X - m[:, np.newaxis, np.newaxis, np.newaxis]
                d = np.std(X.numpy(), axis=(1, 2, 3))
                X = X / d[:, np.newaxis, np.newaxis, np.newaxis]
            else:
                for idx in range(3):
                    m = np.mean(X[:, :, :, idx].numpy(), axis=(1, 2))
                    X[:, :, :, idx] = X[:, :, :, idx] - m[:, np.newaxis, np.newaxis]
                    d = np.std(X[:, :, :, idx].numpy(), axis=(1, 2))
                    X[:, :, :, idx] = X[:, :, :, idx] / d[:, np.newaxis, np.newaxis]

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
        sys.stdout.flush()

def train_robust(loader, model, opt, epsilon, epoch, log, verbose, standarization, color,
                real_time=False, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.train()

    end = time.time()
    for i, (X,y) in enumerate(loader):

        if color:
            X = X.permute(0, 2, 3, 1)

        if standarization==True:
            if not color:
                m = np.mean(X.numpy(), axis=(1, 2, 3))
                X = X - m[:, np.newaxis, np.newaxis, np.newaxis]
                d = np.std(X.numpy(), axis=(1, 2, 3))
                X = X / d[:, np.newaxis, np.newaxis, np.newaxis]
            else:
                for idx in range(3):
                    m = np.mean(X[:, :, :, idx].numpy(), axis=(1, 2))
                    X[:, :, :, idx] = X[:, :, :, idx] - m[:, np.newaxis, np.newaxis]
                    d = np.std(X[:, :, :, idx].numpy(), axis=(1, 2))
                    X[:, :, :, idx] = X[:, :, :, idx] / d[:, np.newaxis, np.newaxis]

        if color:
            X = torch.flatten(X, start_dim=1)

        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)
        data_time.update(time.time() - end)

        with torch.no_grad():
            out = model(Variable(X))
            ce = nn.CrossEntropyLoss()(out, Variable(y))
            err = (out.max(1)[1] != y).float().sum()  / X.size(0)


        robust_ce, robust_err = robust_loss(model, epsilon,
                                             Variable(X), Variable(y),
                                             **kwargs)
        opt.zero_grad()
        robust_ce.backward()


        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        opt.step()

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, robust_ce.detach().item(),
                robust_err, ce.item(), err.item(), file=log)

        if verbose and (i % verbose == 0 or real_time):
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.3f} ({rerrors.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.3f} ({errors.avg:.3f})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, errors=errors,
                   rloss = robust_losses, rerrors = robust_errors), end=endline)
        log.flush()
        sys.stdout.flush()
        del X, y, robust_ce, out, ce, err, robust_err
    print('')
    torch.cuda.empty_cache()

def train(config):
    # Setting up training parameters
    args = cArgs(batch_size=config['training_batch_size'], epochs=config['max_num_training_steps'],
                 opt='adam', verbose=200, starting_epsilon=config['epsilon']/10, epsilon=config['epsilon'], lr=config['initial_learning_rate'])

    if config["skip"]:
        print("SKIP")
        return

    backbone_name = config['backbone']
    robust_training = config['robust_training']

    if os.path.isfile(config["model_dir"] + '/results/training.done') and not config["restart"]:
        print("Already trained")
        return

    model_dir = config['model_dir']

    if not os.path.exists(model_dir + '/checkpoints/'):
        os.makedirs(model_dir + '/checkpoints/')
        os.makedirs(model_dir + '/results/')

    print("saving file to {}".format(config["model_dir"]))
    setproctitle.setproctitle(config["model_dir"])
    train_log = open(config["model_dir"] + "/train.log", "w")
    test_log = open(config["model_dir"] + "/test.log", "w")


    # Setting up the data and the model
    if config['data_set'] == 'mnist':

        train_loader, _ = mnist_loaders(args.batch_size)
        _, test_loader = mnist_loaders(args.test_batch_size)
        color = False

    elif config['data_set'] == 'fashion':

        train_loader, _ = fashion_mnist_loaders(args.batch_size)
        _, test_loader = fashion_mnist_loaders(args.test_batch_size)
        color = False

    elif config['data_set'] == 'cifar':

        train_loader, _ = cifar_loaders(args.batch_size)
        _, test_loader = cifar_loaders(args.test_batch_size)
        color = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for X, y in train_loader:
        break
    kwargs = args2kwargs(args, X=Variable(X.cuda()))
    best_err = 1

    if color:
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

    # Main training loop
    if args.opt == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError("Unknown optimizer")

    epochs = int( float(config['max_num_training_steps']) / (60000.0 / float(config['training_batch_size'])) )
    training_time_history = []
    print("TOTAL EPOCHS:" + str(epochs))

    eps_schedule = np.linspace(args.starting_epsilon,
                                args.epsilon,
                                epochs//2 + 1)
    print("epsilon schedule = ")
    print(eps_schedule)
    print('lr=')
    print(args.lr)
    for t in range(epochs):
        epsilon = args.epsilon

        if t < len(eps_schedule) and config['epsilon_scheduler']:
            epsilon = float(eps_schedule[t])
        else:
            epsilon = args.epsilon

        print("epsilon = " + str(epsilon))

        training_time = 0.0
        start = timer()
        # standard training
        if config["robust_training"]:
            train_robust(train_loader, model, opt, epsilon, t,
                         train_log, args.verbose, config['standarize'], color, args.real_time,
                         norm_type=args.norm_train, bounded_input=True,  **kwargs)
        else:
            train_baseline(train_loader, model, opt, t, train_log, args.verbose, config['standarize'], color)

        end = timer()
        training_time += end - start

        print('    {} examples per second'.format(
            60000.0 / training_time))
        training_time_history.append(60000.0 / training_time)

        torch.save(model.state_dict(), config["model_dir"] + "/checkpoints/checkpoint.pth")


    # Flag the training completed and store the training time profile
    open(model_dir + '/results/training.done', 'w').close()
    with open(model_dir + '/results/training_time.pkl', 'wb') as f:
        pickle.dump(training_time_history, f)

