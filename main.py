import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import os

from EDAN import EDAN
import data_loader

from losses import edl_mse_loss, one_hot_embedding, edl_log_loss, relu_evidence

def load_data(root_path, src, tar, batch_size):
    kwargs = {'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs)
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer, lr_scheduler):
    model.train()
    source_loader, target_train_loader, _ = dataloaders
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len(source_loader)
    print(f"\t lr:{optimizer.param_groups[0]['lr']:.6f}")
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, _ = iter_target.next()
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        optimizer.zero_grad()
        label_source_pred, loss_emmd = model(
            data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        y = one_hot_embedding(label_source, args.nclass)
        y = y.cuda()
        edl_Loss = edl_mse_loss(label_source_pred, y.float(), epoch, args.nclass, 10)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        loss = loss_cls + edl_Loss + args.weight * lambd * loss_emmd

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if i % args.log_interval == 0:
            print(
                f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, '
                f'edl_Loss: {edl_Loss.item():.4f}, loss_emmd: {loss_emmd.item():.4f}')


def test(model, dataloader):
    model.eval()
    correct_pred = {classname: 0 for classname in args.classes}
    total_pred = {classname: 0 for classname in args.classes}
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            evidence = relu_evidence(pred)
            alpha = evidence + 2/args.nclass
            bk = evidence / torch.sum(alpha, dim=1, keepdim=True)
            pred = bk.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for label, prediction in zip(target, pred):
                if label == prediction:
                    correct_pred[args.classes[label]] += 1
                total_pred[args.classes[label]] += 1     
        test_loss /= len(dataloader)
        if args.finetest:
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                            accuracy))
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)')
    return correct


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/home/ceroo/data/office31/')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='amazon')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='webcam')
    parser.add_argument('--nclass', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=10)
    parser.add_argument('--useseed', action='store_true', default=False,
                        help='use seed.')
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2022)
    parser.add_argument('--finetest', action='store_true', default=False,
                        help='Subclass precision.')
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    if args.useseed:
        SEED = args.seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src, args.tar, args.batch_size)
    args.classes = dataloaders[-1].dataset.classes
    print(args.classes)
    model = EDAN(num_classes=args.nclass).cuda()

    correct = 0
    stop = 0

    optimizer = torch.optim.SGD(model.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    for epoch in range(1, args.nepoch + 1):
        stop += 1
        train_epoch(epoch, model, dataloaders, optimizer, lr_scheduler)
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop // 3:
            base_lr = optimizer.param_groups[1]['lr'] / 10.0
        else:
            base_lr = optimizer.param_groups[1]['lr']
        optimizer.param_groups[0]['lr'] = 0.1 * base_lr
        optimizer.param_groups[1]['lr'] = base_lr
        optimizer.param_groups[2]['lr'] = base_lr
        
        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break