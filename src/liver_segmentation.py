import argparse
import time

import cv2
import torch
import pandas as pd
import numpy as np

from conda.exports import get_index
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from model import *

dice_total_list = []
rvd_total_list = []
jaccard_total_list = []
voe_total_list = []
sen_total_list = []
centerlist = []
widthlist = []


def get_data(i):
    import dataset
    imgs = dataset.make_dataset(r"E:/LITS/tumour/%d-%d/test" % (center, width))
    imgx = []
    imgy = []
    for img in imgs:
        imgx.append(img[0])
        imgy.append(img[1])
    return imgx[i], imgy[i]


def train_model(model, criterion, optimizer, dataload, num_epochs=30):
    train_loss = []
    for epoch in range(num_epochs):
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        num_iter = (dt_size - 1) // dataload.batch_size + 1
        with tqdm(total=num_iter, ncols=80, desc="训练epoch %d/%d" % (epoch + 1, num_epochs))as t:
            for x, y in dataload:
                t.set_postfix(loss='{:^7.3f}'.format(epoch_loss))
                step += 1
                inputs = x.to(torch.device)
                labels = y.to(torch.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                t.update()

        torch.save(model.state_dict(), r"E:/LITS/tumour/%d-%d/weights/unet_weights_%d.pth" % (center, width, epoch))
        train_loss.append(epoch_loss)
        train_loss_csv = pd.DataFrame(data=train_loss)
        train_loss_csv.to_csv("E:/LITS/tumour/%d-%d/unet_train_loss.csv" % (center, width), encoding='utf-8',
                              index=False)
        if epoch_loss < 0.1:
            print('Trian_loss' % epoch_loss)
            break
        test(epoch)
    return model


def train():
    print(center, width)
    model = resnet34(3, 1).to(torch.device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset(r"E:/LITS/tumour/%d-%d/train/" % (center, width), transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders)


def test(k):
    args.ckp = r"E:/LITS/tumour/%d-%d/weights/unet_weights_%d.pth" % (center, width, k)
    model = resnet34(3, 1).to(torch.device)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = LiverDataset(r"E:/LITS/tumour/%d-%d/test/" % (center, width), transform=x_transforms,
                                 target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()

    with torch.no_grad():
        i = 0
        rvd_total = 0
        dice_total = 0
        jaccard_total = 0
        sen_total = 0
        voe_total = 0
        num = len(dataloaders)
        for x, _ in tqdm(dataloaders, ncols=80, desc="epoch %d" % k):
            x = x.to(torch.device)
            y = model(x)
            img_y = torch.squeeze(y).cpu().numpy()
            mask_path = get_data(i)[1]
            liver = cv2.imread(get_data(i)[0], 0)
            tumour = cv2.imread(get_data(i)[1], 0)

            body = cv2.imread("E:/LITS/tumour/%d-%d/test/%d_body.png" % (center, width, i), 0)
            cv2.imwrite("./result/%d.png" % i, img_y * 255)
            seg = cv2.imread("./result/%d.png" % i, 0)
            ret, seg = cv2.threshold(seg, 128, 255, cv2.THRESH_OTSU)
            cv2.imwrite("./result/%d_b.png" % i, seg)

            dice, sen, rvd = get_index(mask_path, seg)
            rvd_total += rvd
            dice_total += dice
            sen_total += sen
            jaccard = dice / (2 - dice)
            jaccard_total += jaccard
            voe = 1 - jaccard
            voe_total += voe

            if i < num: i += 1
        image_mask = draw_mask_edge_on_image_cv2(body, tumour, seg, color1=(0, 255, 0), color2=(255, 0, 0))
        print('epoch %d - Dice:%f - RVD:%f - Jaccard:%f - Voe:%f - TPR:%f' % (
            k, dice_total / num, rvd_total / num, jaccard_total / num, voe_total / num, sen_total / num))
        plt.imshow(image_mask)
        plt.pause(0.01)
        plt.show()
        val(dice_total / num, rvd_total / num, jaccard_total / num, voe_total / num, sen_total / num)
        time.sleep(1)
        return model


def draw_mask_edge_on_image_cv2(image, mask, seg, color1, color2):
    coef = 255 if np.max(image) < 3 else 1
    image_mask = (image * coef).astype(np.float32)
    contours1, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_mask, contours1, -1, color1, 2)
    cv2.drawContours(image_mask, contours2, -1, color2, 2)
    image_mask = image_mask / 255
    return image_mask


def val(x, y, z, p, q):
    dice_total_list.append(x)
    rvd_total_list.append(y)
    jaccard_total_list.append(z)
    voe_total_list.append(p)
    sen_total_list.append(q)
    dice_total_csv = pd.DataFrame(data=dice_total_list)
    dice_total_csv.to_csv("E:/LITS/tumour/%d-%d/index/unet_dice_loss.csv" % (center, width), encoding='utf-8',
                          index=False)
    rvd_total_csv = pd.DataFrame(data=rvd_total_list)
    rvd_total_csv.to_csv("E:/LITS/tumour/%d-%d/index/unet_rvd_loss.csv" % (center, width), encoding='utf-8',
                         index=False)
    jaccard_total_csv = pd.DataFrame(data=jaccard_total_list)
    jaccard_total_csv.to_csv("E:/LITS/tumour/%d-%d/index/unet_jaccard_loss.csv" % (center, width), encoding='utf-8',
                             index=False)
    voe_total_csv = pd.DataFrame(data=voe_total_list)
    voe_total_csv.to_csv("E:/LITS/tumour/%d-%d/index/unet_voe_loss.csv" % (center, width), encoding='utf-8',
                         index=False)
    sen_total_csv = pd.DataFrame(data=sen_total_list)
    sen_total_csv.to_csv("E:/LITS/tumour/%d-%d/index/unet_sen_loss.csv" % (center, width), encoding='utf-8',
                         index=False)


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])

    y_transforms = transforms.ToTensor()

    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test", default="train")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_known_args()[0]

    for i in range():
        for j in range():
            center = centerlist[i]
            width = widthlist[j]
            train()
