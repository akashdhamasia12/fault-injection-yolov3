from __future__ import division

# from models import *
from darknet import Darknet
from utils.utils import *
from utils.datasets import *
from pytorchfi.core import fault_injection as pfi_core
from pytorchfi.errormodels import single_bit_flip_func as pfi_core_func
from pytorchfi.errormodels import (
    random_inj_per_layer,
    random_inj_per_layer_batched,
    random_neuron_inj,
    random_neuron_inj_batched,
    random_neuron_single_bit_inj,
    random_neuron_single_bit_inj_batched,
    random_neuron_single_bit_inj_layer,
    random_neuron_multi_bit_inj,
    random_weight_single_bit_inj,
    random_weight_multi_bit_inj,
    random_weight_single_bit_inj_layer,
    random_weight_location,
)

import configuration

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data//samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config//yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights//yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data//coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    configuration.init()


    # os.makedirs("output", exist_ok=True)

    # Set up model
    # model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    # model.eval()  # Set in evaluation mode

    configuration.fault_injection = True
    configuration.ranger_on = True
    configuration.normalalize_ranger = False
    configuration.clipping = False
    configuration.layer_specific = False
    configuration.grid_specific = False
    configuration.save_faulty_parameters = True
    configuration.load_faulty_parameters = False
    configuration.exponent_bits_only = False
    configuration.multiple_locations = 1
    configuration.type_fault_injection = "w"
    configuration.which_run = 1000
    configuration.which_bit = 1
    configuration.layer_no = -1

    #Set up the neural network
    print("Loading network.....")
    model = Darknet(opt.model_def)
    model.load_weights(opt.weights_path)
    print("Network successfully loaded")
    # model.eval()  # Set in evaluation mode

    img_size = 416
    batch_size = 1


#############################################
    p = pfi_core_func(
        model,
        img_size,
        img_size,
        batch_size,
        use_cuda=False, #False,
        bits=32,
        bit_loc=-1,
        locations=configuration.multiple_locations,
    )

    ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453] #for Alexnet -> not using as of now


    # inj_model = random_neuron_inj(p, min_val=10000, max_val=20000)

    # inj_model.eval()
    # conv_i = 1
    # k = 15
    # c_i = 20
    # h_i = 2
    # w_i = 3
    # inj_model = random_neuron_inj(p, min_val=10000, max_val=20000)
    inj_model = random_neuron_single_bit_inj(p, ranges)

    inj_model.eval()

    # inj_value_i = 10000.0

    # inj_model = p.declare_weight_fi(
    #     conv_num=conv_i, k=k, c=c_i, h=h_i, w=w_i, value=inj_value_i
    # )

    # inj_model = p.declare_weight_fi(rand=True, min_rand_val=0.0, max_rand_val=10000.0)

    # inj_model.eval()

    # inj_model.eval()
################################################

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    Tensor = torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):

        # # inj_model = random_neuron_inj(p, min_val=10000, max_val=20000)
        # inj_model = random_neuron_single_bit_inj(p, ranges)

        # inj_model.eval()

        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            # detections = model(input_imgs)
            detections = inj_model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        filename = path.split("\\")[-1].split(".")[0]
        # plt.savefig(f"{filename}_faulty.png", bbox_inches="tight", pad_inches=0.0)
        plt.savefig("data//samples//output//" + str(img_i) + "_ranger_faulty.png", bbox_inches="tight", pad_inches=0.0)

        plt.close()
