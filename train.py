# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import os
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from data.dataset import TSNDataSet
from utils.transforms import *
from utils.logger import setup_logger
from utils.lr_scheduler import get_scheduler
from utils.utils import reduce_tensor, set_random_seed
from config import parser
from opts import dataset_config
from model.FLN import FLN
from utils.utils import AverageMeter, accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import *
import torchvision
from datetime import datetime
import wandb
import warnings
from model.resnet import set_parameter_requires_grad
from model.LabelSmoothing import LSR

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    ddp_setup(local_rank=local_rank)

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(
        args.dataset, args.modality)
    full_arch_name = args.arch
    args.store_name = '_'.join(['segment%d'%args.num_segments,
                                'length%d'%args.length, 'step%d'%args.img_step, full_arch_name, args.consensus_type,  'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)

    if dist.get_rank() == 0:
        check_rootfolders()

    logger = setup_logger(output=os.path.join(args.root_log, args.
                        store_name),distributed_rank=dist.get_rank(),
                          name=f'FLN')
    logger.info('storing name: ' + args.store_name)

    set_random_seed(324)

    model = FLN(num_class,
                args.num_segments,
                args.modality,
                base_model=args.arch,
                new_length=args.length,
                img_step=args.img_step,
                dropout=args.dropout,
                img_feature_dim = args.img_feature_dim,
                pretrain=args.pretrain,
                logger=logger)
    
    if dist.get_rank() == 0:
        init_wandb(args.store_name, cfg=args, resume= (True if args.resume is not None else False))
        wandb.watch(model, log='all', log_freq=100, log_graph=False)

    
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset else True)

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    train_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.train_list,
        num_segments=args.num_segments,
        new_length=args.length,
        img_step=args.img_step,
        modality=args.modality,
        image_tmpl=prefix,
        transform=torchvision.transforms.Compose([train_augmentation,
                                                  ToTorchFormatTensor(
                                                      div=True),
                                                  normalize,
                                                 Flatten([args.img_feature_dim,args.img_feature_dim], length=16)]),
        dense_sample=args.dense_sample)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)

    val_dataset = TSNDataSet(
        args.dataset,
        args.root_path,
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        image_tmpl=prefix,
        new_length=args.length,
        img_step=args.img_step,
        random_shift=False,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            GroupScale(scale_size), GroupCenterCrop(crop_size),
            ToTorchFormatTensor(div=True),normalize,Flatten([args.img_feature_dim,args.img_feature_dim], length=16)]),
        dense_sample=args.dense_sample)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, num_workers=args.workers,
                                             pin_memory=True, sampler=val_sampler, drop_last=True)

    # define loss function (criterion) and optimizer
    criterion = list()
    if args.loss_type == 'nll':
        criterion.append(LSR().cuda())
        criterion.append(torch.nn.MSELoss().cuda())
    else:
        raise ValueError("Unknown loss type")
    
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

    scheduler = get_scheduler(optimizer, len(train_loader), args)

    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank],
                                     broadcast_buffers=True, find_unused_parameters=True)

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        logger.info(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                logger.info('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                logger.info('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        logger.info(
            '#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            logger.info('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    if args.evaluate:
        logger.info(("===========evaluate==========="))
        val_loader.sampler.set_epoch(args.start_epoch)
        prec1, prec5, val_loss = validate(val_loader, model, criterion, logger)
        if dist.get_rank() == 0:
            is_best = prec1 > best_prec1
            best_prec1 = prec1
            logger.info(("Best Prec@1: '{}'".format(best_prec1)))

        return

    latest_loss = 4
    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch=epoch, latest_loss=latest_loss, logger=logger, scheduler=scheduler)
        latest_loss = train_loss
        if dist.get_rank() == 0:
            wandb.log({
                    "LR": optimizer.param_groups[0]['lr'],
                    "Loss_train": train_loss,
                    "train_top1": train_top1,
                    "train_top5": train_top5,
                    "epoch": epoch
                })

        # if epoch > 2:
        #     model1 = model.module
        #     set_parameter_requires_grad(model1, freezing=False)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loader.sampler.set_epoch(epoch)
            prec1, prec5, val_loss = validate(
                val_loader, model, criterion, logger)
            latest_loss = val_loss
            if args.lr_scheduler == "reduce" and "SGD" in str(type(optimizer)):
                scheduler.step(metrics=latest_loss)
            if dist.get_rank() == 0:
                wandb.log({
                    "Loss_test": val_loss,
                    "test_top1": prec1,
                    "test_top5": prec5,
                    "epoch": epoch
                })

                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                logger.info(("Best Prec@1: '{}'".format(best_prec1)))
                save_epoch = epoch + 1
                if is_best:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'prec1': prec1,
                            'best_prec1': best_prec1,
                        }, save_epoch, is_best)

    wandb.finish()
    dist.destroy_process_group()

def train(train_loader, model, criterion, optimizer, epoch, latest_loss, beta=0, logger=None, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    class_losses = AverageMeter()
    ind_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acces = AverageMeter()

    losses.avg = latest_loss

    model.train()

    end = time.time()
    for i, (input, target, ind_label) in enumerate(train_loader):

        data_time.update(time.time() - end)
        target = target.cuda()
        ind_label = ind_label.cuda()
        input_var = input.cuda()
        target_var = target
        output, ind_output = model(input_var)
        class_loss = criterion[0](output, target_var)
        if ind_output is not None:
            ind_loss = criterion[1](ind_output, ind_label) * beta / args.num_segments
        else:
            ind_loss = torch.tensor(0).cuda()
        loss = class_loss + ind_loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        if ind_output is not None:
            acc = get_acc(ind_output, ind_label)
        else:
            acc = torch.tensor(0)

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        class_losses.update(class_loss.item(), input.size(0))
        ind_losses.update(ind_loss.item(), input.size(0))
        acces.update(acc.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        if args.lr_scheduler == "reduce" and "SGD" in str(type(optimizer)):
            scheduler.step(metrics=latest_loss)
        else:
            scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i % args.print_freq == 0 and i != 0) or i == len(train_loader)-1:
            logger.info(('Epoch: [{0}/{1}][{2}/{3}],\t lr: {lr:.6f}\t'
                         'Time {batch_time.avg:.3f}\t'
                         'Data {data_time.avg:.3f}\t'
                         'Loss {loss.avg:.3f}\t'
                         'LossInd {ind_loss.avg:.3f}\t'
                         'AccInd {acces.avg:.3f}\t'
                         'Prec@1 {top1.avg:.3f}\t'
                         'Prec@5 {top5.avg:.3f}'.format(
                             epoch, args.epochs, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, ind_loss=ind_losses, acces=acces,
                             top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))  # TODO
            
    return losses.avg, top1.avg, top5.avg



def validate(val_loader, model, criterion, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    ind_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, ind_label) in enumerate(val_loader):
            target = target.cuda()
            ind_label = ind_label.cuda()
            output, ind_output = model(input)
            if ind_output is not None:
                ind_loss = criterion[1](ind_output, ind_label)
            else:
                ind_loss = torch.tensor(0).cuda()
            loss = criterion[0](output, target) + ind_loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            loss = reduce_tensor(loss)
            # ind_loss = reduce_tensor(ind_loss)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)

            losses.update(loss.item(), input.size(0))
            ind_losses.update(ind_loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0 and i != 0) or i == len(val_loader):
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Time  {batch_time.avg:.3f}\t'
                     'Loss  {loss.avg:.4f}\t'
                     'Loss_ind {ind_loss.avg:.4f}\t'
                     'Prec@1 {top1.avg:.3f}\t'
                     'Prec@5 {top5.avg:.3f}'.format(
                         i, len(val_loader), batch_time=batch_time, loss=losses, ind_loss=ind_losses, top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                 .format(top1=top1, top5=top5, loss=losses)))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, epoch, is_best):
    filename = '%s/%s/%d_epoch_ckpt.pth.tar' % (
        args.root_model, args.store_name, epoch)
    torch.save(state, filename)
    if is_best:
        best_filename = '%s/%s/best.pth.tar' % (
            args.root_model, args.store_name)
        torch.save(state, best_filename)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)

def ddp_setup(local_rank):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # rank 0 process
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
    # nccl：NVIDIA Collective Communication Library
    # 分布式情况下的，gpus 间通信
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    print(f"[init] == local rank: {int(os.environ['LOCAL_RANK'])} ==")

def init_wandb(exp_name, cfg, resume):
    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(
        entity="324",  # wandb上对应的team名称（必填）
        project="flatten",  # 本次的项目名称（必填）
        name=f"{exp_name}-{cur_time}",  # 本次实验的名称（可选，如果不设置，wandb会自动生成本次实验名称）
        # tags=["yolo", "lanes-detection"],  # 本次实验的标签（可选）
        notes=f"{exp_name}",  # 本次实验的备注（可选）
        config=cfg.__dict__,  # 本次实验的配置说明（可选）
        resume= resume
    )

def get_acc(input, label):
    acces = (1 - (input.detach() - label.detach()).abs()
               ).sum(0) / args.batch_size
    acc = acces.sum(0) / acces.shape[0]
    return acc

if __name__ == '__main__':
    main()
