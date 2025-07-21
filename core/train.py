import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda
from tensorboardX import SummaryWriter
from warmupMultiStepLR import WarmupMultiStepLR

def train_step(step, optimizer, model, match_loss, data, scheduler):
    model.train()

    # if step == 100000 + 1:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr']/6.

    res_logits, res_e_hat = model(data)
    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()
   # for name, param in model.named_parameters():
        #if torch.any(torch.isnan(param.grad)):
            #print('skip because nan')
           # return loss_val

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return loss_val,loss_i


def train(model, train_loader, valid_loader, config):
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    scheduler = config.scheduler
    scheduler = WarmupMultiStepLR(optimizer, [200000, 400000], warmup_iters=100000,
                                  warmup_factor=0.01, warmup_method='linear')
    # scheduler = WarmupMultiStepLR(optimizer, [200000, 400000], warmup_iters=100000, warmup_factor=0.01,
    #                               warmup_method='linear')
    match_loss = MatchLoss(config)

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    writer = SummaryWriter(os.path.join(config.SummaryWriter_base, config.SummaryWriter_floder))
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss']*(config.iter_num+3))
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])
    train_loader_iter = iter(train_loader)
    for step in trange(start_epoch, config.train_iter, ncols=config.tqdm_width):
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']
        loss_vals, loss = train_step(step, optimizer, model, match_loss, train_data, scheduler)  # 训练
        # loss_vals, loss= train_step(step, optimizer, model, match_loss, train_data,scheduler)   #训练

        writer.add_scalar('Train-Learn Rate', cur_lr, step)
        writer.add_scalar('Train-TotalLoss', loss, step)
        writer.add_scalar('Train-RegressionLoss_Geo_Loss_', loss_vals[0], step)
        writer.add_scalar('Train-ClassifyLoss', loss_vals[1], step)
        writer.add_scalar('Train-L2 Loss', loss_vals[2], step)


        logger_train.append([cur_lr] + loss_vals)

        # Check if we want to write validation
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            va_res, geo_loss, cla_loss, l2_loss,  precision, _, _  = valid(valid_loader, model, step, config)   # 验证

            writer.add_scalar('Val_mAP', va_res, step)
            writer.add_scalar('Val_GeoLoss', geo_loss, step)
            writer.add_scalar('Val_ClaLoss', cla_loss, step)
            writer.add_scalar('Val_L2Loss', l2_loss, step)

            logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
            if va_res> best_acc:
                # 输出Saving best model with va_res, geo_loss, cla_loss, l2_loss,  precision
                print("Saving best model with va_res = {}, geo_loss = {}, cla_loss = {}, l2_loss = {},  precision = {}".format(va_res, geo_loss, cla_loss, l2_loss,  precision))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best1.pth'))
                if precision>0.63:
                    torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': va_res,
                    'optimizer' : optimizer.state_dict(),
                    }, os.path.join(config.log_path, f'model_best{str(precision)}.pth'))
            if step+1 > 300000:
                torch.save({
                    'epoch': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': va_res,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(config.log_path, f'model_best{str(precision)}.pth'))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)

