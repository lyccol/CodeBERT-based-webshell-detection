import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.utils.data.distributed
from matplotlib import pyplot as plt

import NNModel
from NNModel import BERTClass
from dataSet import PhpDataset
import transformers.trainer
import torch.nn.functional as F
import psutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

blue = lambda x: '\033[94m' + x + '\033[0m'
loss_list = []
acc_list = []
loss_list_val = []
acc_list_val = []
batch_size = 32

def detect_cpu_mem():
    """检测CPU和内存占用率"""
    print("进行mem和cpu检测:")
    # 内存检测
    mem = psutil.virtual_memory().percent
    # psutil检测cpu时间隔至少3s以上
    cpu = psutil.cpu_percent(3)
    print("当前内存占用率:" + str(mem) + "%")
    print("当前CPU占用率:" + str(cpu) + "%")
    return  mem, cpu


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    # V = inputs.size().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / 2)


if __name__ == '__main__':
    dataset = PhpDataset()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)


    length = len(dataset)
    train_size, validate_size = int(0.8 * length), int(0.2 * length)
    train_set, validate_set = torch.utils.data.random_split(dataset, [train_size+1, validate_size])
    print([train_size, validate_size])

    test_set, validate_set = torch.utils.data.random_split(validate_set, [656, 655])

    print('train_size:', len(train_set))
    print('test_set:', len(test_set))
    print('validate_set:', len(validate_set))

    training_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(dataset=validate_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    testing_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # dist.init_process_group(backend='nccl')

    gpus= [0, 1, 2, 3, 4, 5, 6, 7]
    # gpus = [0]

    print("model initing")
    # model = BERTClass()
    # model = NNModel.CodeBERTClassifer()
    model = NNModel.TextCNNClassifer()
    model.cuda()
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    # nn.TransformerDecoder


    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)



    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    print("training!!!")

    num_batch = len(train_set) / batch_size
    # print(len(train_set))
    # print(num_batch)

    def train(epoch):
        model.train()
        for _, data in enumerate(training_loader, 0):

            targets = data['targets']
            targets = targets.view(-1, 1)
            targets = torch.LongTensor(targets)
            targets = torch.zeros(batch_size, 2).scatter_(1, targets, 1)

            ids = data['ids'].to(device, dtype=torch.long).cuda(non_blocking=True)
            mask = data['mask'].to(device, dtype=torch.long).cuda(non_blocking=True)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long).cuda(non_blocking=True)
            targets = targets.to(device, dtype=torch.float).cuda(non_blocking=True)
            targets = label_smoothing(targets)
            # print(targets)
            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            # print(outputs)
            loss = loss_fn(outputs, targets)
            # print(outputs, targets)
            # loss.backward()

            # loss_list.append(loss.cpu().detach().numpy().tolist())
            # acc_list.append()
            # print(loss_list)
            pred_choice = outputs.max(1)[1]
            targets = targets.max(1)[1]
            # print(pred_choice, targets)
            correct = pred_choice.eq(targets).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, _, num_batch, loss.item(), correct.item() / float(batch_size)))
            loss_list.append(loss.item())
            acc_list.append(correct.item() / float(batch_size))

            # print(_)
            if _ % 10 == 0:
                # print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                # detect_cpu_mem()
                j, data = next(enumerate(val_loader, 0))
                # print(j, data)
                targets = data['targets']
                targets = targets.view(-1, 1)
                targets = torch.LongTensor(targets)
                targets = torch.zeros(batch_size, 2).scatter_(1, targets, 1)

                ids = data['ids'].to(device, dtype=torch.long).cuda(non_blocking=True)
                mask = data['mask'].to(device, dtype=torch.long).cuda(non_blocking=True)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long).cuda(non_blocking=True)
                target = targets.to(device, dtype=torch.float).cuda(non_blocking=True)
                pred = model(ids, mask, token_type_ids)
                # print(pred, target)

                loss = loss_fn(pred, target)
                pred_choice = pred.max(1)[1]
                target = target.max(1)[1]
                correct = pred_choice.eq(target).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, _, num_batch, blue('val'), loss.item(), correct.item() / float(batch_size)))
                # loss_list_val = []
                # acc_list_val = []
                loss_list_val.append(loss.item())
                acc_list_val.append(correct.item() / float(batch_size))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        torch.save(model.module.state_dict(), '%s/cls_model_%d.pth' % ('model', epoch))

    def validation(epoch):
        model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                targets = data['targets']
                targets = targets.view(-1, 1)
                targets = torch.LongTensor(targets)
                targets = torch.zeros(batch_size, 2).scatter_(1, targets, 1)

                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)



                outputs = model(ids, mask, token_type_ids)
                outputs = outputs.max(1)[1]
                targets = targets.max(1)[1]
                # print(outputs)
                # print(targets)

                # fin_targets.extend(targets.cpu().detach().numpy().tolist())
                # fin_outputs.extend(torch.sigmoid(outputs.float()).cpu().detach().numpy().tolist())
        return outputs.cpu(), targets.cpu()

    for epoch in range(10):
        train(epoch)
        for epoch in range(1):
            outputs, targets = validation(epoch)
            outputs = np.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
            f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
            print(f"Accuracy Score = {accuracy}")
            print(f"F1 Score (Micro) = {f1_score_micro}")
            print(f"F1 Score (Macro) = {f1_score_macro}")

    print(loss_list)
    print(acc_list)
    print(loss_list_val)
    print(acc_list_val)


    # torch.load()




    for epoch in range(1):
        outputs, targets = validation(epoch)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")