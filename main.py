from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

'''
def SpecifiedLabel(OriginalLabel):
    targetlabel = OriginalLabel + 1
    targetlabel = targetlabel % 10
    return targetlabel
'''
GPU ='4,5'
os.environ['CUDA_VISIBLE_DEVICES'] =GPU
parser = argparse.ArgumentParser(
    description='Pytorch Implement Protection for IP of DNN with CIRAR10')
parser.add_argument('--dataset', default='cifar10', help='mnist|cifar10')
parser.add_argument('--dataroot', default='./data/')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=100) # 100 for cifar10    30 for mnist
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--wm_num', nargs='+', default=[500, 600],  # 1% of train dataset, 500 for cifar10, 600 for mnist
                        help='the number of wm images')
parser.add_argument('--wm_batchsize', type=int, default=20, help='the wm batch size')
parser.add_argument('--lr', nargs='+', default=[0.001, 0.1]) # 0.001 for adam    0.1 for sgd
parser.add_argument('--hyper-parameters',  nargs='+', default=[3, 5, 1, 0.1])
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--seed', default=32, type=int,
                    help='seed for initializing training.')
parser.add_argument('--pretrained', type=bool,
                    default=False, help='use pre-trained model')
parser.add_argument('--wm_train', type=bool, default=True,
                    help='whther to watermark  pre-trained model')
args = parser.parse_args()

if torch.cuda.is_available():
    cudnn.benchmark = True
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        '''
        warnings.warn('You have cho5sen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        '''
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from models import SSIM 
from models import *
from models.Discriminator import DiscriminatorNet, DiscriminatorNet_mnist
from models.HidingUNet import UnetGenerator, UnetGenerator_mnist
# save code each time
if args.train:
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.save_path += cur_time + '/'
    os.makedirs(args.save_path+'images', exist_ok=True)
    os.makedirs(args.save_path+'checkpiont', exist_ok=True)
    os.makedirs(args.save_path+'models', exist_ok=True)
    os.mknod(args.save_path + "models/main.py")
    os.mknod(args.save_path + "models/HidingUNet.py")
    os.mknod(args.save_path + "models/Discriminator.py")
    shutil.copyfile('main.py', args.save_path + "models/main.py")
    shutil.copyfile('models/HidingUNet.py',
                    args.save_path + 'models/HidingUNet.py')
    shutil.copyfile('models/Discriminator.py',
                    args.save_path + 'models/Discriminator.py')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Preparing Data
print('==> Preparing data..')
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load trainset and testset
    trainset = torchvision.datasets.CIFAR10(
        root=args.dataroot, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)
    testset = torchvision.datasets.CIFAR10(
        root=args.dataroot, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)
    # load the 1% origin sample
    trigger_set = torchvision.datasets.CIFAR10(
        root=args.dataroot, train=True, download=True, transform=transform_test)
    trigger_loader = torch.utils.data.DataLoader(
        trigger_set, batch_size=args.wm_batchsize, shuffle=False, num_workers=2, drop_last=True)

    # load logo
    ieee_logo = torchvision.datasets.ImageFolder(
        root=args.dataroot+'/IEEE', transform=transform_test)
    ieee_loader = torch.utils.data.DataLoader(ieee_logo, batch_size=1)
    for _, (logo, __) in enumerate(ieee_loader):
        secret_img = logo.expand(
            args.wm_batchsize, logo.shape[1], logo.shape[2], logo.shape[3]).cuda()
elif args.dataset == 'mnist':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    # load trainset and testset
    trainset = torchvision.datasets.MNIST(
        root=args.dataroot, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)
    testset = torchvision.datasets.MNIST(
        root=args.dataroot, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)
    # load the 1% origin sample 
    trigger_set = torchvision.datasets.MNIST(
        root=args.dataroot, train=True, download=True, transform=transform)
    trigger_loader = torch.utils.data.DataLoader(
        trigger_set, batch_size=args.wm_batchsize, shuffle=False, num_workers=2, drop_last=True)
    # load logo
    for _, (logo, l) in enumerate(testloader):
        for k in range(args.batchsize):
            if l[k].cpu().numpy() == 1:
                logo = logo[k:k+1]
                break
        secret_img = logo.expand(args.wm_batchsize, logo.shape[1], logo.shape[2], logo.shape[3]).cuda()
        break

# get the watermark-cover images foe each batch
wm_inputs, wm_cover_labels = [], []
#wm_labels = []
if args.wm_train:
    for wm_idx, (wm_input, wm_cover_label) in enumerate(trigger_loader):
        wm_input, wm_cover_label = wm_input.cuda(), wm_cover_label.cuda()
        wm_inputs.append(wm_input)
        wm_cover_labels.append(wm_cover_label)
        #wm_labels.append(SpecifiedLabel(wm_cover_label))

        if args.dataset == 'cifar10' and wm_idx == (int(args.wm_num[0]/args.wm_batchsize)-1):  # choose 1% of dataset as origin sample
            break
        elif args.dataset == 'mnist' and wm_idx == (int(args.wm_num[1]/args.wm_batchsize)-1):
            break
# Adversarial ground truths

valid = torch.cuda.FloatTensor(args.wm_batchsize, 1).fill_(1.0)
fake = torch.cuda.FloatTensor(args.wm_batchsize, 1).fill_(0.0)
if args.dataset == 'cifar10':
    np_labels = np.random.randint(
        10, size=(int(args.wm_num[0]/args.wm_batchsize), args.wm_batchsize))
elif args.dataset == 'mnist':
    np_labels = np.random.randint(
        10, size=(int(args.wm_num[1]/args.wm_batchsize), args.wm_batchsize))
wm_labels = torch.from_numpy(np_labels).cuda()

#wm_labels = SpecifiedLabel()
best_real_acc, best_wm_acc, best_wm_input_acc = 0, 0, 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss, test_loss = [[], []], [[], []]
train_acc, test_acc = [[], []], [[], []]

# Model
print('==> Building model..')
if args.dataset == 'mnist':
    Hidnet = UnetGenerator_mnist()
    Disnet = DiscriminatorNet_mnist()
elif args.dataset == 'cifar10':
    Hidnet = UnetGenerator()
    Disnet = DiscriminatorNet()

#Dnnet = LeNet5()
Dnnet = VGG('VGG19')
#Dnnet = ResNet101()
#Dnnet = PreActResNet18()
#Dnnet = GoogLeNet()
#Dnnet = MobileNetV2()
#Dnnet = DPN26()


Hidnet = nn.DataParallel(Hidnet.cuda())
Disnet = nn.DataParallel(Disnet.cuda())
Dnnet = nn.DataParallel(Dnnet.cuda())

criterionH_mse = nn.MSELoss()
criterionH_ssim = SSIM()
optimizerH = optim.Adam(Hidnet.parameters(), lr=args.lr[0], betas=(0.5, 0.999))
schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

criterionD = nn.BCELoss()
optimizerD = optim.Adam(Disnet.parameters(), lr=args.lr[0], betas=(0.5, 0.999))
schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=8, verbose=True)

criterionN = nn.CrossEntropyLoss()
optimizerN = optim.SGD(Dnnet.parameters(), lr=args.lr[1], momentum=0.9, weight_decay=5e-4)
schedulerN = MultiStepLR(optimizerN, milestones=[40, 80], gamma=0.1)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    Dnnet.train()
    Hidnet.train()
    Disnet.train()
    wm_cover_correct, wm_correct, real_correct, wm_total, real_total = 0, 0, 0, 0, 0
    loss_H_ = AverageMeter()
    loss_D_ = AverageMeter()
    real_acc = AverageMeter()
    wm_acc = AverageMeter()
    for batch_idx, (input, label) in enumerate(trainloader):
        input, label = input.cuda(), label.cuda()
        wm_input = wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]
        wm_label = wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]
        wm_cover_label = wm_cover_labels[(wm_idx + batch_idx) % len(wm_inputs)]
        #############Discriminator##############
        optimizerD.zero_grad()
        wm_img = Hidnet(wm_input, secret_img)
        wm_dis_output = Disnet(wm_img.detach())
        real_dis_output = Disnet(wm_input)
        loss_D_wm = criterionD(wm_dis_output, fake)
        loss_D_real = criterionD(real_dis_output, valid)
        loss_D = loss_D_wm + loss_D_real
        loss_D.backward()
        optimizerD.step()
        ################Hidding Net#############
        optimizerH.zero_grad()
        optimizerD.zero_grad()
        optimizerN.zero_grad()
        wm_dis_output = Disnet(wm_img)
        wm_dnn_output = Dnnet(wm_img)
        loss_mse = criterionH_mse(wm_input, wm_img)
        loss_ssim = criterionH_ssim(wm_input, wm_img)
        loss_adv = criterionD(wm_dis_output, valid)
      
        loss_dnn = criterionN(wm_dnn_output, wm_label)
        loss_H = args.hyper_parameters[0] * loss_mse + args.hyper_parameters[1] * (1-loss_ssim) + args.hyper_parameters[2] * loss_adv + args.hyper_parameters[3] * loss_dnn
        loss_H.backward()
        optimizerH.step()
        ################DNNet#############
        optimizerN.zero_grad()
        inputs = torch.cat([input, wm_img.detach()], dim=0)
        labels = torch.cat([label, wm_label], dim=0)
        dnn_output = Dnnet(inputs)
      
        loss_DNN = criterionN(dnn_output, labels)
        loss_DNN.backward()
        optimizerN.step()

        # calculate the accuracy
        wm_cover_output = Dnnet(wm_input)
        _, wm_cover_predicted = wm_cover_output.max(1)
        wm_cover_correct += wm_cover_predicted.eq(wm_cover_label).sum().item()

        _, wm_predicted = dnn_output[args.batchsize: args.batchsize +
                                     args.wm_batchsize].max(1)
        wm_correct += wm_predicted.eq(wm_label).sum().item()
        wm_total += args.wm_batchsize

        _, real_predicted = dnn_output[0:args.batchsize].max(1)
        real_correct += real_predicted.eq(
            labels[0:args.batchsize]).sum().item()
        real_total += args.batchsize

        print('[%d/%d][%d/%d]  Loss D: %.4f Loss_H: %.4f (mse: %.4f ssim: %.4f adv: %.4f)  Loss_real_DNN: %.4f Real acc: %.3f  wm acc: %.3f' % (
            epoch, args.num_epochs, batch_idx, len(trainloader),
            loss_D.item(), loss_H.item(), loss_mse.item(
            ), loss_ssim.item(), loss_adv.item(), loss_DNN.item(),
            100. * real_correct / real_total, 100. * wm_correct / wm_total))

        loss_H_.update(loss_H.item(), int(input.size()[0]))
        loss_D_.update(loss_D.item(), int(input.size()[0]))
        real_acc.update(100. * real_correct / real_total)
        wm_acc.update(100. * wm_correct / wm_total)
    train_loss[0].append(loss_H_.avg)
    train_loss[1].append(loss_D_.avg)
    train_acc[0].append(real_acc.avg)
    train_acc[1].append(wm_acc.avg)
    save_loss_acc(epoch, train_loss, train_acc, True)


def test(epoch):
    Dnnet.eval()
    Hidnet.eval()
    Disnet.eval()
    global best_real_acc
    global best_wm_acc
    global best_wm_input_acc
    wm_cover_correct, wm_correct, real_correct, real_total, wm_total = 0, 0, 0, 0, 0
    Hlosses = AverageMeter()  
    Dislosses = AverageMeter()  
    real_acc = AverageMeter()
    wm_acc = AverageMeter()
    DNNlosses = AverageMeter()
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(testloader):
            input, label = input.cuda(), label.cuda()
            wm_input = wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_label = wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_cover_label = wm_cover_labels[(
                wm_idx + batch_idx) % len(wm_inputs)]
            #############Discriminator###############
            wm_img = Hidnet(wm_input, secret_img)
            wm_dis_output = Disnet(wm_img.detach())
            real_dis_output = Disnet(wm_input)
            loss_D_wm = criterionD(wm_dis_output, fake)
            loss_D_real = criterionD(real_dis_output, valid)
            loss_D = loss_D_wm + loss_D_real
            Dislosses.update(loss_D.item(), int(wm_input.size()[0]))

            ################Hidding Net#############
            wm_dnn_outputs = Dnnet(wm_img)
            loss_mse = criterionH_mse(wm_input, wm_img)
            loss_ssim = criterionH_ssim(wm_input, wm_img)
            loss_adv = criterionD(wm_dis_output, valid)
         
            loss_dnn = criterionN(wm_dnn_outputs, wm_label)
            loss_H = args.hyper_parameters[0] * loss_mse + args.hyper_parameters[1] * (1-loss_ssim) + args.hyper_parameters[2] * loss_adv + args.hyper_parameters[3] * loss_dnn
            Hlosses.update(loss_H.item(), int(input.size()[0]))
            ################DNNet#############
            inputs = torch.cat([input, wm_img.detach()], dim=0)
            labels = torch.cat([label, wm_label], dim=0)
            dnn_outputs = Dnnet(inputs)
        
            loss_DNN = criterionN(dnn_outputs, labels)
            DNNlosses.update(loss_DNN.item(), int(inputs.size()[0]))

           
            wm_cover_output = Dnnet(wm_input)
            _, wm_cover_predicted = wm_cover_output.max(1)
            wm_cover_correct += wm_cover_predicted.eq(
                wm_cover_label).sum().item()

            #wm_dnn_output = Dnnet(wm_img)
            # _, wm_predicted = wm_dnn_output.max(1)
            _, wm_predicted = dnn_outputs[args.batchsize:
                                          args.batchsize + args.wm_batchsize].max(1)
            wm_correct += wm_predicted.eq(wm_label).sum().item()
            wm_total += args.wm_batchsize

            _, real_predicted = dnn_outputs[0:args.batchsize].max(1)
            real_correct += real_predicted.eq(
                labels[0:args.batchsize]).sum().item()
            real_total += args.batchsize

    val_hloss = Hlosses.avg
    val_disloss = Dislosses.avg
    val_dnnloss = DNNlosses.avg
    real_acc.update(100. * real_correct / real_total)
    wm_acc.update(100. * wm_correct / wm_total)
    test_acc[0].append(real_acc.avg)
    test_acc[1].append(wm_acc.avg)
    print('Real acc: %.3f  wm acc: %.3f wm cover acc: %.3f ' % (
        100. * real_correct / real_total, 100. * wm_correct / wm_total, 100. * wm_cover_correct / wm_total))

    resultImg = torch.cat([wm_input, wm_img, secret_img], 0)
    torchvision.utils.save_image(resultImg, args.save_path + 'images/Epoch_' + str(epoch) + '_img.png', nrow=args.wm_batchsize,
                                 padding=1, normalize=True)
    test_loss[0].append(val_hloss)
    test_loss[1].append(val_disloss)

    save_loss_acc(epoch, test_loss, test_acc, False)
    # save
    real_acc = 100. * real_correct / real_total
    wm_acc = 100. * wm_correct / wm_total
    wm_input_acc = 100. * wm_cover_correct / wm_total
    if real_acc >= best_real_acc:  # and (wm_acc >= best_wm_acc):
        print('Saving...')

        Hstate = {
            'net': Hidnet.module if torch.cuda.is_available() else Hidnet,
            'epoch': epoch,
        }
        Dstate = {
            'net': Disnet.module if torch.cuda.is_available() else Disnet,
            'epoch': epoch,
        }
        Nstate = {
            'net': Dnnet.module if torch.cuda.is_available() else Dnnet,
            'acc': real_acc,
            'wm_acc': wm_acc,
            # 'wm_labels':np_labels,
            'epoch': epoch,
        }

        torch.save(Hstate, args.save_path + 'checkpiont/Hidnet.pt')
        torch.save(Dstate, args.save_path + 'checkpiont/Disnet.pt')
        torch.save(Nstate, args.save_path + 'checkpiont/Dnnet.pt')
        best_real_acc = real_acc
    if wm_acc > best_wm_acc:
        best_wm_acc = wm_acc

    if wm_input_acc > best_wm_input_acc:
        best_wm_input_acc = wm_input_acc
    return val_hloss, val_disloss, val_dnnloss, best_real_acc, best_wm_acc, best_wm_input_acc


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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


def save_loss_acc(epoch, loss, acc, train):


    _, ax1 = plt.subplots()  
    ax2 = ax1.twinx()  

    ax1.plot(np.arange(epoch+1), loss[0], '-y', label='ste-model loss')
    ax1.plot(np.arange(epoch+1), loss[1], '-r', label='discriminator loss')
    ax2.plot(np.arange(epoch+1), acc[0], '-g', label='real_acc')
    ax2.plot(np.arange(epoch+1), acc[1], '-b', label='wm_acc')

    ax1.set_xlabel('Epoch(' + ",".join(str(l) for l in args.hyper_parameters) + ')')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')

    ax1.set_ylim(0, 5)
    ax2.set_ylim(0, 100)

    ax1.legend(loc=1)
    ax2.legend(loc=2)
    if train:
        plt.savefig(args.save_path + 'results_train_'+GPU+'.png')
    else:
        plt.savefig(args.save_path + 'results_test_'+GPU+'.png')
    plt.close()


for epoch in range(args.num_epochs):
    train(epoch)
    val_hloss, val_disloss, val_dnnloss, acc, wm_acc, wm_inut_acc = test(epoch)
    schedulerH.step(val_hloss)
    schedulerD.step(val_disloss)
    schedulerN.step()

