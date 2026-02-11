from data_utils.scanobjectnn import get_sets
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from pointnet_cls import get_model, get_loss  # pointnet
import sys
import provider
import random


def get_version():
    str1 = 'Python  version : {}'.format(sys.version.replace('\n', ' '))
    str2 = "PyTorch version : {}".format(torch.__version__)
    str3 = "cuda   version : {}".format(torch.version.cuda)
    return [str1, str2, str3]


def seed_torch(random_seed):
    seed = int(random_seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--data_root', type=str, default='/media/one/系统/chenzhuang/Datasets/h5_files/main_split_nobg')
    parser.add_argument('--num_class', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    parser.add_argument('--model', default='pointnet_cls', help='model name')
    parser.add_argument('--task', type=str, default='origin', help='task for create the log')

    parser.add_argument('--refine_time', type=int, default=3)
    parser.add_argument('--weight', type=float, default=0.5)

    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training')
    parser.add_argument('--step_size',  default=20, type=int)
    parser.add_argument('--gamma', default=0.7, type=float)

    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Sampled Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    """we have set the channel of the point coluds to 3 which is normal"""
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    return parser.parse_args()


def test(classifier, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """set the random seed"""
    seed_torch(args.random_seed)

    '''CREATE DIR'''
    '''2023-08-12_14-31'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    '''./表示同一级目录下，./可以不写'''
    """experiment_dir文件夹下存放两个文件，一个是日志text文件，另一个是.pth文件，
    所有迭代过程中在验证集上效果最好的模型"""
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)  # 创建log目录（exist_ok:只有当目录不存在时创建目录）
    experiment_dir = experiment_dir.joinpath(args.task)
    experiment_dir.mkdir(exist_ok=True)  # 在log目录下创建task(classification)目录
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)  # 日志等级大于等于Info的都会输出
    # 设置每行日志前面的时间
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (experiment_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    '''Hypeparamter LOADING'''
    logger.info('####Parameters####')
    logger.info(args)
    '''Version Information Store'''
    logger.info('####Versions####')
    logger.info(get_version())
    '''DATA LOADING'''
    logger.info('####Load dataset####')

    trainDataLoader, testDataLoader = get_sets(args.data_root, args.batch_size)

    '''MODEL LOADING'''
    classifier = get_model(args.num_class, normal_channel=args.normal)
    criterion = get_loss(weight=args.weight)  # using own-defined loss function
    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    try:  # 如果存在已经训练好的模型(之前没有训练完的模型)，直接加载
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0  # best result on test dataset
    mean_correct = []  # average accuracy on train dataset

    '''TRANING'''
    logger.info('####Start training####')
    for epoch in range(start_epoch, args.epoch):  # [0, 200]
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            # 随机对点进行删除，变换以及移动等操作
            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            """数据集中的数据是按照 batch_size;n_points;channel形式组织的；模型中需要按照
            batch_size;channel;n_points的形式进行组织"""
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()  # target:torch.Size([8, 1])
            target = target[:, 0]  # # target:torch.Size([8])

            optimizer.zero_grad()
            # 因为有dropout层，和BN层，需要调整为train模型
            classifier = classifier.train()
            pred, features = classifier(points)  # batch_size;num_class(8*40); feature_shape:B*F*P
            loss = criterion(pred, target.long(), features, args.refine_time)  # 参数维度不一样，顺序重要
            # 按行找寻最大值（pred.data.max(1)）；pred.data.max(1)[0]:行最大值的数值
            # pred.data.max(1)[1]：行最大值的下标
            pred_choice = pred.data.max(1)[1]  # torch.Size([8])
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)
        print('Train Instance Accuracy: %f' % train_instance_acc)

        """after training one epoch, test its performance on test dataset"""
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=args.num_class)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            print('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            # save the best model
            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(experiment_dir) + '/best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': best_instance_acc,
                    'class_acc': best_class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)