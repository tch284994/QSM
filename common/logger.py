r""" Logging during training/testing """
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import torch


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, dataset):
        self.benchmark = dataset.benchmark
        if self.benchmark == 'pascal':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 20
        elif self.benchmark == 'fss':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 1000
        elif self.benchmark == 'deepglobe':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 6
        elif self.benchmark == 'isic':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 3
        elif self.benchmark == 'lung':
            self.class_ids_interest = dataset.class_ids
            self.class_ids_interest = torch.tensor(self.class_ids_interest).cuda()
            self.nclass = 1

        self.intersection_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.union_buf = torch.zeros([2, self.nclass]).float().cuda()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = torch.stack(self.loss_buf)
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch

        msg += 'mIoU: %5.2f ' % iou
        msg += 'FB-IoU: %5.2f | ' % fb_iou
        msg += 'Avg L: %6.5f  ' % loss_buf.mean()

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] | ' % (batch_idx+1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f ' % loss_buf[-1]
                msg += 'Avg L: %6.5f ' % loss_buf.mean()
            msg += 'mIoU: %5.2f ' % iou
            msg += 'FB-IoU: %5.2f | ' % fb_iou

            # 追加时间信息
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            msg += 'time: %s' % formatted_time

            Logger.info(msg)


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):

        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')  # get the current time
        logpath = args.logpath if training else args.logpath+ '_TEST' + logtime
        if logpath == '': logpath = logtime  # When the log file path is empty, build a folder by the current time

        cls.logpath = os.path.join('logs', logpath + '.log')  # Splicing the complete log file path
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath)  # create a directory

        # Write logs to both file and console
        logging.basicConfig(filemode='w',  # write mode
                            filename=os.path.join(cls.logpath, 'log.txt'),  # Set the path and name of the log file
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Create a SummaryWriter object of Tensorboard, which is used to record the indicators in the training process
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Record the key parameters of training/testing
        logging.info('\n:============== CD-FSS with TCHNet ==============')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':=================================================\n')



    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0  # 3145728 (ACAM params)

        for k in model.state_dict().keys():

            # print(k)
            # if k == 'g1':
            #     Logger.info('The value of g1 is: %f ' % model.state_dict()[k])

            if k == 'g2':
                Logger.info('The value of g1 is: %f ' % float(2 - float(model.state_dict()[k])))
                Logger.info('The value of g2 is: %f ' % model.state_dict()[k])

            # if k == 'g3':
            #     Logger.info('The value of g3 is: %f ' % model.state_dict()[k])

            if k == 'g4':
                Logger.info('The value of g3 is: %f ' % float(2 - float(model.state_dict()[k])))
                Logger.info('The value of g4 is: %f ' % model.state_dict()[k])

            n_param = model.state_dict()[k].view(-1).size(0)
            if k.split('.')[0] in 'backbone':
                if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param

        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))

