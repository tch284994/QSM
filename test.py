import torch
from common.logger import Logger, AverageMeter
from model.QSM import QSM

import torch.nn as nn
from data_util.datasets import FSSDataset
from common import utils
import argparse
from common.evaluation import Evaluator
from common.vis import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Domain-Rectifying Adapter for CD-FSS')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--nshot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    # you should change "n-shot" during test

    # Test dataset path
    parser.add_argument('--benchmark', type=str, default='deepglobe', choices=['deepglobe', 'isic', 'lung', 'fss'], help='The benchmark to be tested')
    parser.add_argument('--test_datapath', type=str, default='../data/deepglobe', help='The path to the benchmark dataset')
    # ###select from {../data/deepglobe, ../data/ISIC, ../data/chest, ../data/fss}

    # Model loading path
    parser.add_argument('--checkpoint_path', type=str, default='./outdir/QSM_pth/QSM_best.pth', help='Checkpoint path')
    # ###select from {./outdir/QSM_pth/QSM_best.pth, ./outdir/QSM_pth/best_model.pth, ./outdir/QSM_pth/last_model.pth}

    # Visualization Path
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--vis_path', type=str, default='./vis_result/deepglobe/', help='vis path')
    # you should change "vis_path" during test

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

        M2_f = 0  # final mask
        voted_masks = []  # Store the mask obtained from each prediction
        for i in range(nshot):
            sup_rgb = batch['support_imgs'][:,i,:]
            sup_msk = batch['support_masks'][:,i,:]
            # [sup_rgb][0].shape: [1, 3, 400, 400]
            # [sup_msk][0].shape: [1, 400, 400]
            qry_rgb = batch['query_img']
            qry_msk = batch['query_mask']

            pred, M_self = model([sup_rgb], [sup_msk], qry_rgb, qry_msk, training=False)
            M2 = pred[0]
            M2_f += M2.argmax(dim=1)  # Used for subsequent voting
            voted_masks.append(M_self)
            if nshot == 1: break  # Return the current prediction result

            # qry_msk.shape:          [1, 400, 400]
            # pred.shape:        [bsz, 2, 400, 400]
            # M_self.shape:         [bsz, 400, 400]
            # logit_mask_agg.shape: [bsz, 400, 400]

        # Perform nshot prediction on M2_f
        bsz = M2_f.size(0)
        max_vote = M2_f.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)  # Get maximum value
        M2_f = M2_f.float() / max_vote           # Calculate the voting ratio for each pixel point
        M2_f[M2_f < 0.5] = 0                     # background
        M2_f[M2_f >= 0.5] = 1                    # foreground
        assert M2_f.size() == batch['query_mask'].size()

        # voted_masks = torch.stack(voted_masks)          # [nshot, bsz, 400, 400]
        # M_self = torch.mode(voted_masks, dim=0).values  # Vote to determine pixel category

        # Evaluate prediction
        # area_inter, area_union = Evaluator.classify_prediction(M_self.clone(), batch)  # Evaluate M_self
        area_inter, area_union = Evaluator.classify_prediction(M2_f.clone(), batch)      # Evaluate M2_f
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=100)

        # ### M_self Visualizer
        # if Visualizer.visualize:
        #     Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
        #                                           batch['query_img'], batch['query_mask'],
        #                                           M_self, batch['class_id'], idx,
        #                                           area_inter[1].float() / area_union[1].float())

        # ### M2_f Visualizer
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  M2_f, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


def main(args):
    Logger.initialize(args, training=False)

    # initialize
    Evaluator.initialize()
    Visualizer.initialize(args.vis_path,args.visualize)

    FSSDataset.initialize(img_size=400, datapath=args.test_datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, 1, 0, 'test', args.nshot)

    Logger.info(f'     ==> {len(dataloader_test)} testing samples')
    dim_ls = [128, 256, 512]
    model = QSM(args.backbone,dim_ls)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint,strict=False)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    Evaluator.initialize()

    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')


if __name__ == '__main__':

    args = parse_args()
    main(args)
