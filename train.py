import argparse
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import common
import dataset
import model


def trainer(train, model, optimizer, lossfunc):
    print("---------- Start Training ---------")

    try:
        with tqdm(trainloader, ncols=100) as pbar:
            train_loss = 0.0
            out_list = list()
            for images, labels in pbar:
                images, labels = Variable(images), Variable(labels)
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = lossfunc(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        return train_loss

    except ValueError:
        pass

def validater(test, model):
    print("---------- Start Testing ---------")

    try:
        valid_loss = 0.0
        with tqdm(testloader, ncols=100) as pbar:
            for images, labels in pbar:
                images, labels = Variable(images), Variable(labels)
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = lossfunc(outputs, labels)
                valid_loss += loss.item()

        return valid_loss

    except ValueError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--network", required=True,
                        help="choose network, only ResNet50 or ResNet50LSTM")
    parser.add_argument("--input_ch", type=int, required=True,
                        help="the number of input channel")
    parser.add_argument("--batch_size", type=int, default=1, required=False,
                        help="adapt batch size to machine's memory")
    parser.add_argument("--early_stopping", required=False,
                        help="if early stopping, add this argument")
    parser.add_argument("--checkpoint", required=False,
                        help="if retry learning, write model path")
    args = parser.parse_args()

    print("---------- Loading Data ----------")
    datas = dataset.setup_data()
    print("---------- Finished loading Data ----------")

    # split train and validation dataset and set up model
    train_size = int(round(datas.length * 0.8))
    test_size = datas.length - train_size
    print("train size : ", train_size)
    print("test_size  : ", test_size)
    if args.network == "ResNet50":
        train, test = torch.utils.data.random_split(datas, [train_size, test_size])
        model = model.ResNet50(pretrained=True, num_input_channel=args.input_ch,
                               num_output=2) # output=(x, y)
    elif args.network == "ResNet50LSTM":
        train = torch.utils.data.Subset(datas, list(range(0, train_size)))
        test = torch.utils.data.Subset(datas, list(range(train_size, datas.length)))
        model = model.ResNet50LSTM(pretrained=True, num_input_channel=args.input_ch,
                                   num_output=2) # output=(x, y)
    elif args.network == "ResNet18LSTM":
        train = torch.utils.data.Subset(datas, list(range(0, 80000)))
        test = torch.utils.data.Subset(datas, list(range(80000, 100000)))
        model = model.ResNet18LSTM(pretrained=True, num_input_channel=args.input_ch,
                                   num_output=2) # output=(x, y)
    elif args.network == "LSTM":
        train = torch.utils.data.Subset(datas, list(range(0, train_size)))
        test = torch.utils.data.Subset(datas, list(range(train_size, datas.length)))
        model = model.LSTM(pretrained=True, num_input_channel=args.input_ch,
                           num_output=2, batch_size=args.batch_size) # output=(x, y)

    trainloader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(
            test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # set up GPU
    model, device = common.setup_device(model)

    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # lr=0.001, weight_decay=0.01)
#     lossfunc = nn.CrossEntropyLoss()
    lossfunc = nn.MSELoss()

    # tensorboard
    writer = SummaryWriter(log_dir="./logs")

    # main
    train_loss_list = list()
    valid_loss_list = list()
    early_stopping = [np.inf, 5, 0]
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss_list = checkpoint["loss"]
#         valid_loss_list = checkpoint["valid_loss_list"]
        for i, train_loss in enumerate(train_loss_list):
            writer.add_scalar("Train Loss", train_loss, i+1)
        for i, valid_loss in enumerate(valid_loss_list):
            writer.add_scalar("Valid Loss", valid_loss, i+1)
        print("reload model : ", start_epoch, " and restart")
    for epoch in range(100):
        epoch += start_epoch
        try:
            # train
            train_loss = trainer(trainloader, model, optimizer, lossfunc)
            train_loss_list.append(train_loss)

            # test
            with torch.no_grad():
                valid_loss = validater(testloader, model)
                valid_loss_list.append(valid_loss)

            # show loss and accuracy
            print("%d : train_loss : %.3f" % (epoch + 1, train_loss))
            print("%d : valid_loss : %.3f" % (epoch + 1, valid_loss))

            # save model
            torch.save({
                "epoch" : epoch + 1,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "train_loss_list" : train_loss_list,
                "valid_loss_list" : valid_loss_list
                }, "./models/" + str(epoch + 1))

            # tensorboard
            writer.add_scalar("Train Loss", train_loss, epoch + 1)
            writer.add_scalar("Valid Loss", valid_loss, epoch + 1)

            # early stopping #TODO
            if args.early_stopping:
                if valid_loss < early_stopping[0]:
                    early_stopping[0] = valid_loss
                    early_stopping[-1] = 0
                    torch.save(model.state_dict(), "./models/model.pth")
                else:
                    early_stopping[-1] += 1
                    if early_stopping[-1] == early_stopping[1]:
                        break

        except ValueError:
            continue
