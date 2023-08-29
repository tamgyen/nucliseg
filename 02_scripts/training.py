import math
import sys
import time

import cv2
import numpy as np

from data_loading import EndoNukeDataset, show

import torch
import torch.utils.data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

import lightning as L

import utils


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = utils.get_coco_api_from_dataset(data_loader.dataset)
    iou_types = utils.get_iou_types(model)
    coco_evaluator = utils.CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    #     transforms.append(T.RandomVerticalFlip(0.5))
    transforms.append(T.ToImageTensor())
    transforms.append(T.ConvertImageDtype(torch.float32))
    transforms.append(T.SanitizeBoundingBox())
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))

# class LightningWrappedRCNN(L.LightningModule):
#     def __init__(self, base_model):
#         super().__init__()
#         self.model = base_model
#
#     def training_step(self, batch, batch_idx):
#         images, targets = batch
#
#         loss_dict = model(images, targets)
#         loss = sum(loss for loss in loss_dict.values())
#
#         self.log("train_loss", loss)
#         return loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(device.type)

    # our dataset has two classes only - background and person
    num_classes = 2

    model = get_model_instance_segmentation(num_classes)
    # use our dataset and defined transformations
    dataset = EndoNukeDataset(root='../01_data/01_dataset_endonuke', transforms=get_transform(train=True))
    dataset_test = EndoNukeDataset(root='../01_data/01_dataset_endonuke', transforms=get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-500])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        collate_fn=collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        torch.save(model, f'../03_models/MASK_RCNN_v01_ep_{epoch}.pt')


if __name__ == '__main__':
    # print(torch.cuda.is_available())

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    #
    # dataset = EndoNukeDataset(root='../01_data/01_dataset_endonuke', transforms=get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=2, shuffle=True,
    #     collate_fn=collate_fn)

    # example = next(iter(data_loader))
    # show(example, show_masks=False, show_boxes=True)

    # images, targets = next(iter(data_loader))

    # output = model(images, targets)

    # print(':)')

    #main()

    test_image = cv2.imread('../01_data/src.jpg')
    test_crop = test_image[0:1000, 0:1000, :]
    test_crop = cv2.resize(test_crop, (200, 200))

    # test_example = cv2.imread('C:/Dev/Projects/3DHISTEC/01_data/01_dataset_endonuke/images/2357.png')
    #
    # test_compare = np.zeros((200, 400, 3), dtype='uint8')
    #
    # test_compare[0:200, 0:200, :] = test_crop
    # test_compare[0:200, 200:400, :] = test_example
    #
    # cv2.imshow('im', test_compare.astype('uint8'))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    model = torch.load(f'../03_models/MASK_RCNN_v01_ep_4.pt')

    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_crop_torch = torch.from_numpy(test_crop).float().permute(2, 0, 1)

    x = [test_crop_torch.to(device)]

    preds = model(x)

    show(((test_crop_torch,), preds))

    print(':)')
