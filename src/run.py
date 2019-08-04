import os
import sys

sys.path.append('../')
sys.path.append('../config')
sys.path.append('../model')
sys.path.append('../src')
sys.path.append('../data')

import time
import numpy as np
# import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import model.resnet as resnet
from data.PokerDataSet import PokerDataSet

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '-d',
    '--device',
    type=str,
    default='cuda:0',
    help='using GPU and cuda to train and also indicate the GPU number'
)
parser.add_argument(
    '-sd',
    '--seed',
    type=int,
    default=42,
    help='indicate the seed number while doing random to reach reproductable'
)
parser.add_argument(
    '--batchsize',
    type=int,
    default=128,
    help='batch size for training'
)
parser.add_argument(
    '--train_root',
    type=str,
    default='../data/train',
    help='path to the train data'
)

parser.add_argument(
    '--train_size',
    type=int,
    default=3180066,
    help='size of the trainset'
)
parser.add_argument(
    '--val_root',
    type=str,
    default='../data/val',
    help='path to the val data'
)

parser.add_argument(
    '--val_size',
    type=int,
    default=97128,
    help='size of the valset'
)
parser.add_argument(
    '--test_root',
    type=str,
    default='../data/test',
    help='path to the testset'
)

parser.add_argument(
    '--test_size',
    type=int,
    default=96737,
    help='size of the testset'
)
parser.add_argument(
    '--save_dir',
    type=str,
    default='../expr',
    help='the path to save model'
)
parser.add_argument(
    '--save_model_freq',
    type=int,
    default=2,
    help='Frequency of saving model'
)
args = parser.parse_args()
print(args)


def run():
    # Device CPU / GPU
    if args.device is not None:
        if args.device.startswith('cuda') and torch.cuda.is_available():
            device = torch.device(args.device)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            # logger.warning(
            #     '{} GPU is not available, running on CPU'.format(__name__))
            print(
                'Warning: {} GPU is not available, running on CPU'
                .format(__name__))
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')

    # logger.info('{} Using device: {}'.format(__name__, device))
    print('{} Using device: {}'.format(__name__, device))

    # Seeding
    if args.seed is not None:
        # logger.info('{} Setting random seed'.format(__name__))
        print('{} Setting random seed'.format(__name__))
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Import Dataset and DataLoader
    trainset = PokerDataSet(
        root=args.train_root,
        size=args.train_size
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize,
        shuffle=True, num_workers=4
    )

    valset = PokerDataSet(
        root=args.val_root,
        size=args.val_size
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batchsize,
        shuffle=False, num_workers=4
    )

    testset = PokerDataSet(
        root=args.test_root,
        size=args.test_size
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize,
        shuffle=False, num_workers=4
    )

    net = resnet.resnetpokernet(num_classes=13707).to(device)

    # Define Loss function and optimizer
    # Try to use Cross-Entropy and Adam (or SGD with momentum)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_acc_val = 0.0
    # Train the Network
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the param gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # 200 - train
            running_loss += loss.item()
            if (i + 1) % 200 == 0:
                print(
                    '[%d, %5d] loss: %.3f'
                    % (epoch + 1, i + 1, running_loss / 200)
                )
                running_loss = 0.0

            # 2000 - eval
            if (i + 1) % 2000 == 0:
                correct = 0
                total = 0
                t_init = time.time()
                with torch.no_grad():
                    for data in valloader:
                        states, labels = data[0].to(device), data[1].to(device)
                        outputs = net(states)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                t_final = time.time()
                acc = correct / total
                is_best_acc = acc > best_acc_val
                best_acc_val = max(acc, best_acc_val)
                print(
                    'Making Evaluation on 97128 states, Accuracy: %.3f %%'
                    % (100 * correct / total)
                )
                print(
                    'Time per inference: {} s'
                    .format((t_final - t_init) / total)
                )
                if is_best_acc:
                    best_acc_state = {
                        'state_dict': net.state_dict(),
                        'acc': best_acc_val,
                    }
                    torch.save(
                        best_acc_state,
                        os.path.join(args.save_dir, 'best_acc.pth'))
                    print(
                        'Saved model for best acc {}'
                        .format(best_acc_val)
                    )

        if (epoch + 1) % args.save_model_freq == 0:
            model_state = {
                'state_dict': net.state_dict()
            }
            torch.save(
                model_state, '{0}/crnn_Rec_done_{1}.pth'
                .format(args.save_dir, epoch + 1)
            )

    print('Finished Training !')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            states, labels = data[0].to(device), data[1].to(device)
            outputs = net(states)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            'Making Evaluation on 97128 states, Accuracy: %.3f %%'
            % (100 * correct / total)
        )


if __name__ == '__main__':
    # running the train/val/test
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    run()
