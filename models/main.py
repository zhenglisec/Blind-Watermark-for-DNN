'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau
#from models.vgg import VGG
from models import PreActResNet18, PreActResNet34, PreActResNet50, VGG, GoogLeNet,MobileNet

from models.mymodel import SSIM, ImageFolderCustomClass
from models.mymodel.HidingUNet import UnetGenerator
from models.mymodel.DiscriminatorNet import DiscriminatorNet1
from adabound import AdaBound
import os
import argparse
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

from torchvision import models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

GPU='7'
os.environ['CUDA_VISIBLE_DEVICES']=GPU
parser = argparse.ArgumentParser(description='Pytorch Implement Protection for IP of DNN with CIRAR10')
parser.add_argument('--dataroot', default='./datasets')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=100)
parser.add_argument('--wm_batchsize', default=2, type=int, help='the wm batch size')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alpha', '--list', nargs='+', default=[1])
parser.add_argument('--beta', '--list1', nargs='+', default=[3, 5, 1, 0.1]) #在暂时是最优
parser.add_argument('--wm_path', type=str, default='./datasets/cifar-10-batches-py/triggerset_from_train')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--seed', default=22, type=int, help='seed for initializing training. ')
parser.add_argument('--pretrained', type=bool, default=False, help='use pre-trained model')
parser.add_argument('--wm_train', type=bool, default=True, help='whther to watermark  pre-trained model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
args = parser.parse_args()
'''
# create model
if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
else:
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=10)
'''
'''
Data = np.uint8(np.random.uniform(150, 180, (32, 32, 3)))
transform = transforms.Compose([ transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) ] )
imData = transform(Data)
imData = torch.unsqueeze(imData, dim=0)
'''
#out1 = model.features(imData)
#out2 = model(imData)
'''
if args.arch.find('vgg')!= -1:
    model.classifier[0] = nn.Linear(512, 512, bias=True)
    model.classifier[3] = nn.Linear(512, 512, bias=True)
    model.classifier[6] = nn.Linear(512, 10, bias=True)
elif args.arch.find('resnet')!= -1:
    model.fc = nn.Linear(2028, 10, bias=True)
print(model)

if torch.cuda.is_available():
    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have cho5sen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
'''

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 保存
if args.train:
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.save_path += cur_time +'/'
    os.makedirs(args.save_path+'images', exist_ok=True)
    os.makedirs(args.save_path+'checkpiont', exist_ok=True)
    os.makedirs(args.save_path+'models', exist_ok=True)
    os.makedirs(args.save_path + 'cover_trigger', exist_ok=True)
    os.mknod(args.save_path + "models/main.py")
    os.mknod(args.save_path + "models/hiding.py")
    os.mknod(args.save_path + "models/dis.py")
    shutil.copyfile('main.py', args.save_path + "models/main.py")
    shutil.copyfile('models/mymodel/HidingUNet.py', args.save_path + 'models/hiding.py')
    shutil.copyfile('models/mymodel/DiscriminatorNet.py', args.save_path + 'models/dis.py')
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Data

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(180),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_ieee_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_ieee_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True, download=True, transform=transform_train)
#trainset = torchvision.datasets.ImageFolder(root='./datasets/CINIC-10/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2, drop_last=True)
testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=True, transform=transform_test)
#testset = torchvision.datasets.ImageFolder(root='./datasets/CINIC-10/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2, drop_last=True)
triggerset = torchvision.datasets.ImageFolder(root=args.wm_path, transform=transform_train)
triggerloader = torch.utils.data.DataLoader(triggerset, batch_size=args.wm_batchsize, shuffle=False, num_workers=2, drop_last=True)
ieee_train_logo = torchvision.datasets.ImageFolder(root=args.dataroot+'/IEEE', transform=transforms_ieee_train)
ieee_train_loader = torch.utils.data.DataLoader(ieee_train_logo, batch_size=1)
for _, (logo, __) in enumerate(ieee_train_loader):
    secret_img = logo.expand(args.wm_batchsize, logo.shape[1], logo.shape[2], logo.shape[3]).cuda()

# Adversarial ground truths
valid = torch.cuda.FloatTensor(args.wm_batchsize, 1).fill_(1.0)
fake = torch.cuda.FloatTensor(args.wm_batchsize, 1).fill_(0.0)
np_labels = np.random.randint(10, size=(50,args.wm_batchsize))
wm_labels = torch.from_numpy(np_labels).cuda()

best_real_acc, best_wm_acc = 0, 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_loss, test_loss = [[],[]], [[],[]]
train_acc, test_acc = [[],[]], [[],[]]

# get the watermark images
wm_inputs = []#, []
if args.wm_train:
    for wm_idx, (wm_input, _) in enumerate(triggerloader):
        wm_input = wm_input.cuda()#, wmtarget[wm_idx].cuda()
        wm_inputs.append(wm_input)
        #wmtargets.append(wmtarget)
wm_idx = np.random.randint(len(wm_inputs))

# Model
print('==> Building model..')
Hidnet = UnetGenerator()
Disnet = DiscriminatorNet1()
Dnnet = VGG('VGG11')
#Dnnet = model
# Dnnet = gcv.models.resnet50(pretrained=False)
# net = ResNet18()
#Dnnet = PreActResNet18()
#Dnnet = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
#Dnnet = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
#print(Hidnet)
Hidnet = nn.DataParallel(Hidnet.cuda())
Disnet = nn.DataParallel(Disnet.cuda())
Dnnet = nn.DataParallel(Dnnet.cuda())
#Hidnet.apply(weights_init)
#Disnet.apply(weights_init)
#Dnnet.apply(weights_init)
print(Hidnet)
print(Disnet)
print(Dnnet)
# 定义损失函数和优化器
criterionH_mse = nn.MSELoss()
criterionH_ssim = SSIM()
optimizerH = optim.Adam(Hidnet.parameters(), lr=args.lr, betas=(0.5, 0.999))
schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

criterionD = nn.BCELoss()
optimizerD = optim.Adam(Disnet.parameters(), lr=args.lr, betas=(0.5, 0.999))
schedulerD = ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=8, verbose=True)

criterionN = nn.CrossEntropyLoss()
optimizerN = optim.SGD(Dnnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
schedulerN = MultiStepLR(optimizerN, milestones=[60, 130], gamma=0.1)
#optimizerDNN = AdaBound(Dnnet.parameters(), lr = 0.001, final_lr=0.1, weight_decay=1e-4)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    Dnnet.train()
    Hidnet.train()
    Disnet.train()
    wm_correct, real_correct, wm_total, real_total = 0, 0, 0, 0
    loss_H_ = AverageMeter()
    loss_D_ = AverageMeter()
    real_acc = AverageMeter()
    wm_acc = AverageMeter()
    for batch_idx, (input, label) in enumerate(trainloader):
        input, label = input.cuda(), label.cuda()
        wm_input = wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]
        wm_label = wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]
        #############优化Discriminator###############
        optimizerD.zero_grad()
        wm_img = Hidnet(wm_input, secret_img)
        wm_dis_output = Disnet(wm_img.detach())
        real_dis_output = Disnet(wm_input)
        loss_D_wm = criterionD(wm_dis_output, fake)
        loss_D_real = criterionD(real_dis_output, valid)
        loss_D = loss_D_wm + loss_D_real
        loss_D.backward()
        optimizerD.step()
        ################优化Hidding Net#############
        optimizerH.zero_grad()
        optimizerD.zero_grad()
        optimizerN.zero_grad()
        wm_dis_output = Disnet(wm_img)
        wm_dnn_output = Dnnet(wm_img)
        loss_mse = criterionH_mse(wm_input, wm_img)
        loss_ssim = criterionH_ssim(wm_input, wm_img)
        loss_adv = criterionD(wm_dis_output, valid)
        loss_dnn = criterionN(wm_dnn_output, wm_label)
        loss_H = args.beta[0] * loss_mse + args.beta[1] * (1 - loss_ssim) + args.beta[2] * loss_adv  + args.beta[3] * loss_dnn
        loss_H.backward()
        optimizerH.step()
        ################优化 DNNet#############
        optimizerN.zero_grad()
        inputs = torch.cat([input, wm_input.detach()], dim=0)
        labels = torch.cat([label, wm_label], dim=0)
        dnn_output = Dnnet(inputs)
        loss_DNN = criterionN(dnn_output, labels)
        loss_DNN.backward()
        optimizerN.step()

        _, wm_predicted = wm_dnn_output.max(1)
        _, real_predicted = dnn_output[0:args.batchsize].max(1)
        wm_correct += wm_predicted.eq(wm_label).sum().item()
        real_correct += real_predicted.eq(labels[0:args.batchsize]).sum().item()
        wm_total += args.wm_batchsize
        real_total += args.batchsize

        print('[%d/%d][%d/%d]  Loss D: %.4f () Loss_H: %.4f (mse: %.4f ssim: %.4f adv: %.4f)  Loss_real_DNN: %.4f Real acc: %.3f  wm acc: %.3f GPU: %d' % (
            epoch, args.num_epochs, batch_idx, len(trainloader),
            loss_D.item(), loss_H.item(), loss_mse.item(), loss_ssim.item(), loss_adv.item(), loss_DNN.item(),
             100. * real_correct / real_total, 100. * wm_correct / wm_total, int(GPU)))

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
    wm_correct, real_correct, real_total, wm_total = 0, 0, 0, 0
    Hlosses = AverageMeter()  # 纪录每个epoch H网络的loss
    Dislosses = AverageMeter()  # 纪录每个epoch R网络的loss
    real_acc = AverageMeter()
    wm_acc = AverageMeter()
    DNNlosses = AverageMeter()
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(testloader):
            input, label = input.cuda(), label.cuda()
            '''
            for j in range(args.batchsize):
                torchvision.utils.save_image(inputs[j],
                                         args.save_path + 'cover_trigger/' + str(batch_idx) + '_' + str(j) + '_cover.png',
                                         nrow=1, padding=0, normalize=True)
                ll += 1
                if ll > 1000:
                    exit()
            inputs, targets = inputs.cuda(), targets.cuda()
            '''
            wm_input = wm_inputs[(wm_idx + batch_idx) % len(wm_inputs)]
            wm_label = wm_labels[(wm_idx + batch_idx) % len(wm_inputs)]
            #############优化Discriminator###############
            wm_img = Hidnet(wm_input, secret_img)
            wm_dis_output = Disnet(wm_img.detach())
            real_dis_output = Disnet(wm_input)
            loss_D_wm = criterionD(wm_dis_output, fake)
            loss_D_real = criterionD(real_dis_output, valid)
            loss_D = loss_D_wm + loss_D_real
            Dislosses.update(loss_D.item(), int(wm_input.size()[0]))

            ################优化Hidding Net#############
            wm_dnn_outputs = Dnnet(wm_img)
            loss_mse = criterionH_mse(wm_input, wm_img)
            loss_ssim = criterionH_ssim(wm_input, wm_img)
            loss_adv = criterionD(wm_dis_output, valid)
            loss_dnn = criterionN(wm_dnn_outputs, wm_label)
            loss_H = args.beta[0] * loss_mse + args.beta[1] * (1 - loss_ssim) + args.beta[2] * loss_adv + args.beta[
                3] * loss_dnn
            Hlosses.update(loss_H.item(), int(input.size()[0]))
            ################优化 DNNet#############
            inputs = torch.cat([input, wm_input.detach()], dim=0)
            labels = torch.cat([label, wm_label], dim=0)
            dnn_outputs = Dnnet(inputs)
            loss_DNN = criterionN(dnn_outputs, labels)
            DNNlosses.update(loss_DNN.item(), int(inputs.size()[0]))

            _, wm_predicted = wm_dnn_outputs.max(1)
            _, real_predicted = dnn_outputs[0:args.batchsize].max(1)
            wm_correct += wm_predicted.eq(wm_label).sum().item()
            real_correct += real_predicted.eq(labels[0:args.batchsize]).sum().item()
            wm_total += args.wm_batchsize
            real_total += args.batchsize

    val_hloss = Hlosses.avg
    val_disloss = Dislosses.avg
    val_dnnloss = DNNlosses.avg
    real_acc.update(100. * real_correct / real_total)
    wm_acc.update(100. * wm_correct / wm_total)
    test_acc[0].append(real_acc.avg)
    test_acc[1].append(wm_acc.avg)
    print('Real acc: %.3f  wm acc: %.3f  H_loss: %.3f D_loss: %.3f' % (100. * real_correct / real_total, 100. * wm_correct / wm_total, val_hloss, val_disloss))

    resultImg = torch.cat([wm_input, wm_img, secret_img], 0)
    torchvision.utils.save_image(resultImg, args.save_path + 'images/Epoch_' + str(epoch) + '_img.png', nrow=2,
                                padding=1, normalize=True)
    test_loss[0].append(val_hloss)
    test_loss[1].append(val_disloss)

    save_loss_acc(epoch, test_loss, test_acc, False)
    # 保存模型权重
    real_acc = 100. * real_correct / real_total
    wm_acc = 100. * wm_correct / wm_total
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
            'wm_labels':np_labels,
            'epoch': epoch,
        }

        torch.save(Hstate, args.save_path + 'checkpiont/Hidnet.pt')
        torch.save(Dstate, args.save_path + 'checkpiont/Disnet.pt')
        torch.save(Nstate, args.save_path + 'checkpiont/Dnnet.pt')
        best_real_acc = real_acc
        if wm_acc > best_wm_acc:
            best_wm_acc = wm_acc
    return  val_hloss, val_disloss, val_dnnloss, best_real_acc, best_wm_acc
def SpecifiedLabel(OriginalLabel):
    targetlabel = OriginalLabel + 1
    targetlabel = targetlabel % 5
    return targetlabel
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
    ##绘制loss,acc曲线##

    _, ax1 = plt.subplots()  # 声明一个figure对象(_), 一个axis对象(ax1,对应左Y轴)
    ax2 = ax1.twinx()  # 再声明一个

    ax1.plot(np.arange(epoch+1), loss[0], '-y', label='ste-model loss')
    ax1.plot(np.arange(epoch+1), loss[1], '-r', label='discriminator loss')
    ax2.plot(np.arange(epoch+1), acc[0], '-g', label='real_acc')
    ax2.plot(np.arange(epoch+1), acc[1], '-b', label='wm_acc')

    ax1.set_xlabel('Epoch(' + ",".join(str(i) for i in args.alpha) +'|'+ ",".join(str(l) for l in args.beta) + ')')
    ax1.set_ylabel('Train Loss')
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
    #test(epoch)
    train(epoch)
    val_hloss, val_disloss, val_dnnloss, acc, wm_acc = test(epoch)
    schedulerH.step(val_hloss)
    schedulerD.step(val_disloss)
    #print('D')
    schedulerN.step()
print(acc, wm_acc)