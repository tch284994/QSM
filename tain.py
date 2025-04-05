from model.QSM import QSM
from common import utils
from util.utils import count_params, set_seed, Compute_iou
import argparse
from copy import deepcopy
import os
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from tqdm import tqdm
from data_util.datasets import FSSDataset
from common.logger import Logger, AverageMeter
import torch.nn as nn
from common.evaluation import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Domain-Rectifying Adapter for CD-FSS')
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='batch size of training')  # default = 8
    parser.add_argument('--lr',
                        type=float,
                        default=0.0013,
                        help='learning rate')           # default = 0.0013
    parser.add_argument('--size',
                        type=int,
                        default=400,
                        help='Size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal'],
                        help='training dataset')
    parser.add_argument('--train_datapath',
                        type=str,
                        default='../data/VOCdevkit',
                        help='The path of training dataset')

    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='./outdir/Ori_SSP_trained_on_VOC.pth',
                        help='Checkpoint trained from original SSP that has not involved in adapter')

    parser.add_argument('--test_datapath',
                        type=str,
                        default='../data/chest',
                        help='The path of the benchmark')
    parser.add_argument('--benchmark',
                        type=str,
                        default='lung',
                        help = 'The benchmark to be tested')
    parser.add_argument('--global_noise_std',
                        type=float,
                        default=1.0)
    parser.add_argument('--local_noise_std',
                        type=float,
                        default=0.75)

    # Log save path
    parser.add_argument('--logpath', type=str, default='')
    args = parser.parse_args()
    return args


def test(model, dataloader, nshot):

    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)

        M2_f = 0
        voted_masks = []  # Used to store the mask obtained from each prediction
        for i in range(nshot):
            sup_rgb = batch['support_imgs'][:,i,:]
            sup_msk = batch['support_masks'][:,i,:]
            # [sup_rgb][0].shape: ([1, 3, 400, 400])
            # [sup_msk][0].shape: ([1, 400, 400])

            qry_rgb = batch['query_img']
            qry_msk = batch['query_mask']

            pred, M_self = model([sup_rgb], [sup_msk], qry_rgb, qry_msk, training=False)
            M2 = pred[0]
            M2_f += M2.argmax(dim=1)  # for subsequent voting
            voted_masks.append(M_self)
            if nshot == 1: break  # If nshot = 1, directly return the current prediction result

            # qry_msk.shape:        [1, 400, 400]
            # pred.shape:           [bsz, 2, 400, 400]
            # M_self.shape:         [bsz, 400, 400]
            # logit_mask_agg.shape: [bsz, 400, 400]

        # Perform nshot predictions on the final mask
        bsz = M2_f.size(0)
        max_vote = M2_f.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)  # Get maximum value
        M2_f = M2_f.float() / max_vote                     # Calculate the voting ratio for each pixel point
        M2_f[M2_f < 0.5]  = 0                               # background
        M2_f[M2_f >= 0.5] = 1                              # foreground
        assert M2_f.size() == batch['query_mask'].size()

        # Evaluate final mask
        # area_inter, area_union = Evaluator.classify_prediction(M_self.clone(), batch)  # Evaluate M_self
        area_inter, area_union = Evaluator.classify_prediction(M2_f.clone(), batch)      # Evaluate M2_f
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=10)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


def main():

    save_path = 'outdir/QSM_pth'  # Model save path
    os.makedirs(save_path, exist_ok=True)  # Create the path

    args = parse_args()
    FSSDataset.initialize(img_size=args.size, datapath=args.train_datapath)
    trainloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, 'trn', args.shot)
    FSSDataset.initialize(img_size=args.size, datapath=args.test_datapath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, 15, 0, 'test', args.shot)

    dim_ls = [128, 256, 512]  # The output channels of feature map for the first three stages
    model = QSM(args.backbone,dim_ls,args.local_noise_std)

    x_param_loss=nn.L1Loss()                        # absolute error loss (L1 loss)
    criterion = CrossEntropyLoss(ignore_index=255)  # CrossEntropyLoss (CE loss)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)

    Logger.initialize(args, training=True)
    Logger.log_params(model)

    model = DataParallel(model).cuda()

    previous_best = 0  # Initialize the optimal mIoU value

    datum_center_ls = []  # Style mean initialization
    for epoch in range(5):

        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best mIoU = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        global_noise_ls = []  # The noise during global perturbation: (alpha, beta), needs to be given in advance
        for dim in dim_ls:    # Generate random noise to perturb all channels
            ones_mat = torch.ones((args.batch_size, dim, 1, 1))
            zeros_mat = torch.zeros((args.batch_size, dim, 1, 1))

            # Generate an alpha & beta tensor that follows a (0,1) normal distribution
            alpha = torch.normal(zeros_mat, args.global_noise_std * ones_mat).cuda()
            beta = torch.normal(zeros_mat, args.global_noise_std * ones_mat).cuda()
            pert = (alpha, beta)  # Combine into a tuple
            global_noise_ls.append(pert)
        model.train()

        total_loss = 0.0

        tbar = tqdm(trainloader)  # Monitor the progress of model training through tqdm
        set_seed(args.seed)       # Ensure the reproducibility of the experiment
        for idx, batch in enumerate(tbar):
            sup_rgb = batch['support_imgs'].squeeze().cuda()
            sup_msk = batch['support_masks'].long().squeeze().cuda()
            qry_rgb = batch['query_img'].cuda()
            qry_msk = batch['query_mask'].long().cuda()
            if epoch == 0:             # epoch_0 Used to obtain global style
                with torch.no_grad():
                    out_ls, m_self = model([sup_rgb], [sup_msk], qry_rgb, qry_msk, training=True, get_prot=True)
            else:
                out_ls, m_self = model([sup_rgb], [sup_msk], qry_rgb, qry_msk, training=True,global_noise_ls=global_noise_ls,
                                   datum_center_ls=datum_center_ls)

            mask_s = torch.cat([sup_msk], dim=0)  # Get support mask

            x_style = out_ls[-1]  # Obtain the original style mean and original style variance of the current batch

            #  Global style update: Global style 0.99, current style 0.01
            for num in range(len(x_style)):
                current_center = x_style[num][0].mean(dim=0, keepdim=True)
                current_var = x_style[num][1].mean(dim=0, keepdim=True)
                if len(datum_center_ls)<len(dim_ls):
                    datum_center = current_center
                    datum_var = current_var
                    datum_center_ls.append([datum_center,datum_var])
                else:
                    datum_center_ls[num][0] = datum_center_ls[num][0] * 0.99 + 0.01 * current_center
                    datum_center_ls[num][1] = datum_center_ls[num][1] * 0.99 + 0.01 * current_var

            if epoch==0:
                continue

            # Calculate disturbance loss and correction loss
            # La = L1(miu(F0),miu(Fp)) + L1(sigma(F0),sigma(Fp))
            # Lc = L1(miu(F0),miu(Fr)) + L1(sigma(F0),sigma(Fr))

            x_param_ls = torch.zeros(1).cuda()
            if len(out_ls[-2])>0:  # out_ls[-2]: x_params
                for num in range(len(out_ls[-2])):
                    x_ori_mean, x_ori_var, x_new_mean, x_new_var,x_new_rect_mean,x_new_rect_var = out_ls[-2][num]
                    x_param_ls += (x_param_loss(x_ori_mean,x_new_mean) + x_param_loss(x_ori_var,x_new_var)+
                                  x_param_loss(x_ori_mean,x_new_rect_mean) + x_param_loss(x_ori_var,x_new_rect_var))
                x_param_ls = x_param_ls / len(out_ls[-2])  # Average all correction losses to obtain 1/2 (La+Lc)

            # L_BCE = BCE(M2_q, Mq) + BCE(M_p_q, Mq) + BCE(M_p_s, Ms)  # the self similarity mask after perturbation
            loss = criterion(out_ls[0], qry_msk) + criterion(out_ls[1], qry_msk) + criterion(out_ls[2], mask_s) * 0.2

            # L = L_BCE + La + Lc, Obtain joint losses
            loss += (2 * x_param_ls[0])

            optimizer.zero_grad()  # Gradient reset
            loss.backward()        # backward computing gradient
            optimizer.step()       # Update Parameters

            total_loss += loss.item()

            tbar.set_description('Loss:%.2f' % (total_loss / (idx + 1)))

        if epoch==0:
            continue

        model.eval()
        set_seed(args.seed)
        with torch.no_grad():
            test_miou, test_fb_iou = test(model, dataloader_val, args.shot)
        Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))

        # Save the best model
        if test_miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = test_miou
            torch.save(best_model.module.state_dict(),os.path.join(save_path, 'best_model.pth'))

        # Save the latest model
        torch.save(model.module.state_dict(),os.path.join(save_path, 'last_model.pth'))


if __name__ == '__main__':
    main()
