from __future__ import division

# from models import *
from darknet import Darknet
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

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

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import configuration

# fault_injection = False

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):

    print("configuration.scenario", configuration.scenario)
    print("configuration.fault_injection", configuration.fault_injection)
    print("configuration.ranger_on", configuration.ranger_on)
    print("configuration.normalalize_ranger", configuration.normalalize_ranger)
    print("configuration.clipping", configuration.clipping)
    print("configuration.layer_specific", configuration.layer_specific)
    print("configuration.grid_specific", configuration.grid_specific)
    print("configuration.save_faulty_parameters", configuration.save_faulty_parameters)
    print("configuration.load_faulty_parameters", configuration.load_faulty_parameters)
    print("configuration.multiple_locations", configuration.multiple_locations)
    print("configuration.type_fault_injection", configuration.type_fault_injection)
    print("configuration.exponent_bits_only", configuration.exponent_bits_only)
    print("configuration.num_runs", configuration.num_runs)
    print("configuration.which_run", configuration.which_run)
    print("configuration.which_bit", configuration.which_bit)
    print("configuration.layer_no", configuration.layer_no)
    print("configuration.grid_no", configuration.grid_no)

    if configuration.fault_injection==False:
        model.eval()

    if configuration.fault_injection==True:        
        # p = pfi_core(model,
        #     img_size,
        #     img_size,
        #     batch_size,
        #     use_cuda=False,
        # )
        p = pfi_core_func(
            model,
            img_size,
            img_size,
            batch_size,
            use_cuda=True, #False,
            bits=32,
            bit_loc=-1,
            locations=configuration.multiple_locations,
        )

        ranges = [24.375, 26.375, 13.179688, 3.367188, 3.314453] #for Alexnet -> not using as of now

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    print("length of dataloader= ", len(dataloader.dataset))

    if configuration.percent_images > 0:
        configuration.num_images = int((len(dataloader.dataset))*(configuration.percent_images)/100)

    print("num_images for fault/ranger= ", configuration.num_images)

    if configuration.save_faulty_parameters==True and configuration.load_faulty_parameters==False:

        # if configuration.layer_specific==False and configuration.multiple_locations < 2:
        #     output_file_name = "output//fault_parameters_" + str(configuration.type_fault_injection) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(output_file_name,'w')

        # if configuration.layer_specific==False and configuration.exponent_bits_only == False:
        #     output_file_name = "output//fault_parameters_" + str(configuration.type_fault_injection) + "_mul_" + str(configuration.multiple_locations) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(output_file_name,'w')

        # elif configuration.layer_specific==False and configuration.exponent_bits_only == True:
        #     output_file_name = "output//fault_parameters_exp_1_" + str(configuration.type_fault_injection) + "_mul_" + str(configuration.multiple_locations) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(output_file_name,'w')

        # else: #layer_specific
        #     output_file_name = "output//fault_parameters_" + str(configuration.type_fault_injection) + "_l_" + str(configuration.layer_no) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(output_file_name,'w')

        output_file_name = "output//fault_parameters//fault_parameters_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(configuration.multiple_locations) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(configuration.which_run) + ".txt"
        file_=open(output_file_name,'w')

    if configuration.load_faulty_parameters==True and configuration.save_faulty_parameters==False:

        # if configuration.layer_specific==False and configuration.exponent_bits_only == False:
        #     input_file_name = "output//fault_parameters_" + str(configuration.type_fault_injection) + "_mul_" + str(configuration.multiple_locations) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(input_file_name,'r')

        # elif configuration.layer_specific==False and configuration.exponent_bits_only == True:
        #     input_file_name = "output//fault_parameters_exp_1_" + str(configuration.type_fault_injection) + "_mul_" + str(configuration.multiple_locations) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(input_file_name,'r')

        # else: #layer specific
        #     input_file_name = "output//fault_parameters_" + str(configuration.type_fault_injection) + "_l_" + str(configuration.layer_no) + "_nr_" + str(configuration.which_run) + ".txt"
        #     file_=open(input_file_name,'r')

        input_file_name = "output//fault_parameters//fault_parameters_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(configuration.multiple_locations) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(configuration.which_run) + ".txt"
        file_=open(input_file_name,'r')
        # print(input_file_name)


    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tensor = torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    # conv_, k_, c_in_, kH_, kW_ = 0, 0, 0, 0, 0
    if configuration.scenario == "FW": #Fixed weight fault
        (conv_, k_, c_in_, kH_, kW_) = random_weight_location(p, -1)
        rand_bit = random.randint(0, p.bits - 1)
        p.bit_loc = rand_bit
        print(k_, conv_, c_in_, kH_, kW_, rand_bit)


    for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        if configuration.fault_injection==True and configuration.layer_specific==False: #single bitflip fault injection at ny location

            if configuration.save_faulty_parameters==True and configuration.multiple_locations < 2:

                if configuration.type_fault_injection == "n":
                    # inj_model = random_neuron_inj(p, min_val=-10000, max_val=10000)
                    inj_model = random_neuron_single_bit_inj(p, ranges)
                else: #weight injection

                    if configuration.scenario == "FW": #Fixed weight fault
                        inj_model = p.declare_weight_fi(
                            conv_num=conv_, k=k_, c=c_in_, h=kH_, w=kW_, function=p.single_bit_flip_weight)
                    else:
                        inj_model = random_weight_single_bit_inj(p, ranges)

                inj_model.eval()
                file_.write(str(p.CORRUPT_BATCH) + "," + str(p.CORRUPT_CONV) + "," + str(p.CORRUPT_C) + "," + str(p.CORRUPT_H) + "," + str(p.CORRUPT_W) + "," + str(p.CORRUPT_VALUE) + "," + str(p.bit_loc) + "\n")

            elif configuration.save_faulty_parameters==True and configuration.multiple_locations >= 2:

                if configuration.type_fault_injection == "n":
                    inj_model = random_neuron_multi_bit_inj(p, ranges, configuration.multiple_locations)
                else:
                    inj_model = random_weight_multi_bit_inj(p, ranges, configuration.multiple_locations)

                inj_model.eval()
                for i in range(configuration.multiple_locations):
                    file_.write(str(p.CORRUPT_BATCH[i]) + "," + str(p.CORRUPT_CONV[i]) + "," + str(p.CORRUPT_C[i]) + "," + str(p.CORRUPT_H[i]) + "," + str(p.CORRUPT_W[i]) + "," + str(p.CORRUPT_VALUE) + "," + str(p.bit_loc[i]) + "\n")
                # print(str(p.CORRUPT_BATCH) + "," + str(p.CORRUPT_CONV) + "," + str(p.CORRUPT_C) + "," + str(p.CORRUPT_H) + "," + str(p.CORRUPT_W) + "," + str(p.CORRUPT_VALUE) + "," + str(p.bit_loc) + "\n")

            elif configuration.save_faulty_parameters==False and configuration.multiple_locations < 2:
                line = file_.readline().split(",")
                # print(line[0], ",", line[1], ",", line[2], ",", line[3], ",", line[4], ",", line[6])
                p.bit_loc = int(line[6])

                if configuration.type_fault_injection == "n":
                    inj_model = p.declare_neuron_fi(
                            batch=int(line[0]),
                            conv_num=int(line[1]),
                            c=int(line[2]),
                            h=int(line[3]),
                            w=int(line[4]),
                            function=p.single_bit_flip_signed_across_batch,
                        )
                else:
                    inj_model = p.declare_weight_fi(
                        conv_num=int(line[1]), k=int(line[0]), c=int(line[2]), h=int(line[3]), w=int(line[4]), function=p.single_bit_flip_weight)

                inj_model.eval()

            else: #means configuration.multiple_locations >= 2 & configuration.save_faulty_parameters==False

                i=0
                batch_, conv_num_, c_rand_, h_rand_, w_rand_, bit_locations_ = ([] for i in range(6))

                i=0
                for i in range(configuration.multiple_locations):
                    line = file_.readline().split(",")
                    
                    bit_locations_.append(int(line[6])) #random bit_locations       
                    batch_.append(int(line[0]))
                    conv_num_.append(int(line[1]))
                    c_rand_.append(int(line[2]))
                    h_rand_.append(int(line[3]))
                    w_rand_.append(int(line[4]))

                # print(batch_)
                # print(conv_num_)
                # print(c_rand_)
                # print(h_rand_)
                # print(w_rand_)

                p.bit_loc = bit_locations_

                if configuration.type_fault_injection == "n":
                    inj_model = p.declare_neuron_fi(
                            batch=batch_,
                            conv_num=conv_num_,
                            c=c_rand_,
                            h=h_rand_,
                            w=w_rand_,
                            function=p.single_bit_flip_signed_across_batch,
                        )
                else:
                    inj_model = p.declare_weight_fi(
                        conv_num=conv_num_, k=batch_, c=c_rand_, h=h_rand_, w=w_rand_, function=p.single_bit_flip_weight)


                inj_model.eval()

        if configuration.fault_injection==True and configuration.layer_specific==True: #singlebitflip at specified layer

            if configuration.save_faulty_parameters==True:

                if configuration.grid_specific == True:
                    configuration.layer_no = random.randint(configuration.lower_grid, configuration.upper_grid - 1)

                if configuration.type_fault_injection == "n":
                    inj_model = random_neuron_single_bit_inj_layer(p, configuration.layer_no)
                else: #weight fault injection
                    inj_model = random_weight_single_bit_inj_layer(p, configuration.layer_no)

                inj_model.eval()
                file_.write(str(p.CORRUPT_BATCH) + "," + str(p.CORRUPT_CONV) + "," + str(p.CORRUPT_C) + "," + str(p.CORRUPT_H) + "," + str(p.CORRUPT_W) + "," + str(p.CORRUPT_VALUE) + "," + str(p.bit_loc) + "\n")
                # print(str(p.CORRUPT_BATCH) + "," + str(p.CORRUPT_CONV) + "," + str(p.CORRUPT_C) + "," + str(p.CORRUPT_H) + "," + str(p.CORRUPT_W) + "," + str(p.CORRUPT_VALUE) + "," + str(p.bit_loc) + "\n")

            else: #load parameters
                line = file_.readline().split(",")
                # print(line[0], ",", line[1], ",", line[2], ",", line[3], ",", line[4], ",", line[6])
                p.bit_loc = int(line[6])

                if configuration.type_fault_injection == "n":
                    inj_model = p.declare_neuron_fi(
                            batch=int(line[0]),
                            conv_num=int(line[1]),
                            c=int(line[2]),
                            h=int(line[3]),
                            w=int(line[4]),
                            function=p.single_bit_flip_signed_across_batch,
                        )
                else: #weight fault injection
                    inj_model = p.declare_weight_fi(
                        conv_num=int(line[1]), k=int(line[0]), c=int(line[2]), h=int(line[3]), w=int(line[4]), function=p.single_bit_flip_weight)


                inj_model.eval()

        # print("neuron size", sum(x[0]*x[1]*x[2]*x[3] for x in p.get_output_size()))

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        # targets = targets.to(device)

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
    
            if configuration.fault_injection==False:
                outputs = model(imgs)

            if configuration.fault_injection==True:
                outputs = inj_model(imgs)

            outputs = outputs.cpu()
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        # #Save image
        # if configuration.save_image == True:
        # # Create plot
        #     img = np.array(Image.open(img_path))
        #     plt.figure()
        #     fig, ax = plt.subplots(1)
        #     ax.imshow(img)

        #     # Draw bounding boxes and labels of detections
        #     if outputs is not None:
        #         # Rescale boxes to original image
        #         detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        #         unique_labels = detections[:, -1].cpu().unique()
        #         n_cls_preds = len(unique_labels)
        #         bbox_colors = random.sample(colors, n_cls_preds)
        #         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

        #             print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

        #             box_w = x2 - x1
        #             box_h = y2 - y1

        #             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        #             # Create a Rectangle patch
        #             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        #             # Add the bbox to the plot
        #             ax.add_patch(bbox)
        #             # Add label
        #             plt.text(
        #                 x1,
        #                 y1,
        #                 s=classes[int(cls_pred)],
        #                 color="white",
        #                 verticalalignment="top",
        #                 bbox={"color": color, "pad": 0},
        #             )

        #     # Save generated image with detections
        #     plt.axis("off")
        #     plt.gca().xaxis.set_major_locator(NullLocator())
        #     plt.gca().yaxis.set_major_locator(NullLocator())
        #     # filename = path.split("/")[-1].split(".")[0]
        #     filename = path.split("\\")[-1].split(".")[0]
        #     plt.savefig(f"data//samples//{filename}_faulty.png", bbox_inches="tight", pad_inches=0.0)
        #     plt.close()            

        if batch_i >= configuration.num_images:#23454:
            break

    if configuration.save_faulty_parameters==True:
        file_.close()
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config//yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config//coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights//yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data//coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    configuration.init()
    data_config = parse_data_config(opt.data_config)

    if configuration.dataset == "train":
        valid_path = data_config["train"]
    else:
        valid_path = data_config["valid"]

    class_names = load_classes(data_config["names"])

    # # Initiate model
    # model = Darknet(opt.model_def).to(device)
    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path))

    #Set up the neural network
    # print("Loading network.....")
    # model = Darknet(opt.model_def)
    # model.load_weights(opt.weights_path)
    # print("Network successfully loaded")
    # # model.eval()  # Set in evaluation mode
    # print("Compute mAP...")

    if configuration.execute_everything == False:

        precision, recall, AP, f1, ap_class = evaluate(
            model,
            path=valid_path,
            iou_thres=opt.iou_thres,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=1,
        )

        f=open("output//result.txt",'w')

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
            f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        print(f"mAP: {AP.mean()}")
        f.write("mAP," + str(AP.mean()) + "\n")
        f.close()

    else:

        # #ALL SCENARIOS START

        #no fault
        # configuration.fault_injection = False
        # configuration.ranger_on = False
        # configuration.normalalize_ranger = False
        # configuration.clipping = False
        # configuration.layer_specific = False
        # configuration.grid_specific = False
        # configuration.save_faulty_parameters = False
        # configuration.load_faulty_parameters = False
        # configuration.exponent_bits_only = False
        # configuration.multiple_locations = 1
        # configuration.type_fault_injection = "n"
        # configuration.which_run = 0
        # configuration.which_bit = -1
        # configuration.layer_no = -1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(opt.model_def).to(device)
        model.load_weights(opt.weights_path)

        # precision, recall, AP, f1, ap_class = evaluate(
        #     model,
        #     path=valid_path,
        #     iou_thres=opt.iou_thres,
        #     conf_thres=opt.conf_thres,
        #     nms_thres=opt.nms_thres,
        #     img_size=opt.img_size,
        #     batch_size=1,
        # )

        # output_file_name = "output//results//no_fault_gpu_train.txt"

        # f=open(output_file_name,'w')

        # print("Average Precisions:")
        # for i, c in enumerate(ap_class):
        #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        # print(f"mAP: {AP.mean()}")
        # f.write("mAP," + str(AP.mean()) + "\n")
        # f.close()

        # #with faults
        print("total_no. runs ", configuration.num_runs)
        # for runs in range(100, configuration.num_runs):
        for runs in range(3, configuration.num_runs):

            configuration.scenario = "R" #Random multiple bit faults

            for mul_loc in configuration.fault_rate:

        #         configuration.fault_injection = True
        #         configuration.ranger_on = False
        #         configuration.normalalize_ranger = False
        #         configuration.clipping = False
        #         configuration.layer_specific = False
        #         configuration.grid_specific = False
        #         configuration.save_faulty_parameters = True
        #         configuration.load_faulty_parameters = False
        #         configuration.exponent_bits_only = False
        #         configuration.multiple_locations = mul_loc
        #         configuration.type_fault_injection = "n"
        #         configuration.which_run = runs
        #         configuration.which_bit = -1
        #         configuration.layer_no = -1

        #         del model
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = Darknet(opt.model_def).to(device)
        #         model.load_weights(opt.weights_path)

        #         precision, recall, AP, f1, ap_class = evaluate(
        #             model,
        #             path=valid_path,
        #             iou_thres=opt.iou_thres,
        #             conf_thres=opt.conf_thres,
        #             nms_thres=opt.nms_thres,
        #             img_size=opt.img_size,
        #             batch_size=1,
        #         )

        #         output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #         f=open(output_file_name,'w')

        #         print("Average Precisions:")
        #         for i, c in enumerate(ap_class):
        #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #         print(f"mAP: {AP.mean()}")
        #         f.write("mAP," + str(AP.mean()) + "\n")
        #         f.close()

        # #         #with_Ranger

        #         configuration.fault_injection = True
        #         configuration.ranger_on = True
        #         configuration.normalalize_ranger = False
        #         configuration.clipping = False
        #         configuration.layer_specific = False
        #         configuration.grid_specific = False
        #         configuration.save_faulty_parameters = False
        #         configuration.load_faulty_parameters = True
        #         configuration.exponent_bits_only = False
        #         configuration.multiple_locations = mul_loc
        #         configuration.type_fault_injection = "n"
        #         configuration.which_run = runs
        #         configuration.which_bit = -1
        #         configuration.layer_no = -1

        #         del model
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = Darknet(opt.model_def).to(device)
        #         model.load_weights(opt.weights_path)

        #         precision, recall, AP, f1, ap_class = evaluate(
        #             model,
        #             path=valid_path,
        #             iou_thres=opt.iou_thres,
        #             conf_thres=opt.conf_thres,
        #             nms_thres=opt.nms_thres,
        #             img_size=opt.img_size,
        #             batch_size=1,
        #         )

        #         output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #         f=open(output_file_name,'w')

        #         print("Average Precisions:")
        #         for i, c in enumerate(ap_class):
        #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #         print(f"mAP: {AP.mean()}")
        #         f.write("mAP," + str(AP.mean()) + "\n")
        #         f.close()


        #         #with_Ranger_norm

        #         configuration.fault_injection = True
        #         configuration.ranger_on = True
        #         configuration.normalalize_ranger = True
        #         configuration.clipping = False
        #         configuration.layer_specific = False
        #         configuration.grid_specific = False
        #         configuration.save_faulty_parameters = False
        #         configuration.load_faulty_parameters = True
        #         configuration.exponent_bits_only = False
        #         configuration.multiple_locations = mul_loc
        #         configuration.type_fault_injection = "n"
        #         configuration.which_run = runs
        #         configuration.which_bit = -1
        #         configuration.layer_no = -1

        #         del model
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = Darknet(opt.model_def).to(device)
        #         model.load_weights(opt.weights_path)

        #         precision, recall, AP, f1, ap_class = evaluate(
        #             model,
        #             path=valid_path,
        #             iou_thres=opt.iou_thres,
        #             conf_thres=opt.conf_thres,
        #             nms_thres=opt.nms_thres,
        #             img_size=opt.img_size,
        #             batch_size=1,
        #         )

        #         output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #         f=open(output_file_name,'w')

        #         print("Average Precisions:")
        #         for i, c in enumerate(ap_class):
        #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #         print(f"mAP: {AP.mean()}")
        #         f.write("mAP," + str(AP.mean()) + "\n")
        #         f.close()

                #with_Clipper

                configuration.fault_injection = True
                configuration.ranger_on = True
                configuration.normalalize_ranger = False
                configuration.clipping = True
                configuration.layer_specific = False
                configuration.grid_specific = False
                configuration.save_faulty_parameters = False
                configuration.load_faulty_parameters = True
                configuration.exponent_bits_only = False
                configuration.multiple_locations = mul_loc
                configuration.type_fault_injection = "n"
                configuration.which_run = runs
                configuration.which_bit = -1
                configuration.layer_no = -1

                del model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = Darknet(opt.model_def).to(device)
                model.load_weights(opt.weights_path)

                precision, recall, AP, f1, ap_class = evaluate(
                    model,
                    path=valid_path,
                    iou_thres=opt.iou_thres,
                    conf_thres=opt.conf_thres,
                    nms_thres=opt.nms_thres,
                    img_size=opt.img_size,
                    batch_size=1,
                )

                output_file_name = "output//results//faulty_gpu_clipper_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

                f=open(output_file_name,'w')

                print("Average Precisions:")
                for i, c in enumerate(ap_class):
                    print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
                    f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

                print(f"mAP: {AP.mean()}")
                f.write("mAP," + str(AP.mean()) + "\n")
                f.close()


                #Repeat Same procedure for weight

                # configuration.fault_injection = True
                # configuration.ranger_on = False
                # configuration.normalalize_ranger = False
                # configuration.clipping = False
                # configuration.layer_specific = False
                # configuration.grid_specific = False
                # configuration.save_faulty_parameters = True
                # configuration.load_faulty_parameters = False
                # configuration.exponent_bits_only = False
                # configuration.multiple_locations = mul_loc
                # configuration.type_fault_injection = "w"
                # configuration.which_run = runs
                # configuration.which_bit = -1
                # configuration.layer_no = -1

                # del model
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # model = Darknet(opt.model_def).to(device)
                # model.load_weights(opt.weights_path)

                # precision, recall, AP, f1, ap_class = evaluate(
                #     model,
                #     path=valid_path,
                #     iou_thres=opt.iou_thres,
                #     conf_thres=opt.conf_thres,
                #     nms_thres=opt.nms_thres,
                #     img_size=opt.img_size,
                #     batch_size=1,
                # )

                # output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

                # f=open(output_file_name,'w')

                # print("Average Precisions:")
                # for i, c in enumerate(ap_class):
                #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
                #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

                # print(f"mAP: {AP.mean()}")
                # f.write("mAP," + str(AP.mean()) + "\n")
                # f.close()

        #         #with_Ranger

        #         configuration.fault_injection = True
        #         configuration.ranger_on = True
        #         configuration.normalalize_ranger = False
        #         configuration.clipping = False
        #         configuration.layer_specific = False
        #         configuration.grid_specific = False
        #         configuration.save_faulty_parameters = False
        #         configuration.load_faulty_parameters = True
        #         configuration.exponent_bits_only = False
        #         configuration.multiple_locations = mul_loc
        #         configuration.type_fault_injection = "w"
        #         configuration.which_run = runs
        #         configuration.which_bit = -1
        #         configuration.layer_no = -1

        #         del model
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = Darknet(opt.model_def).to(device)
        #         model.load_weights(opt.weights_path)

        #         precision, recall, AP, f1, ap_class = evaluate(
        #             model,
        #             path=valid_path,
        #             iou_thres=opt.iou_thres,
        #             conf_thres=opt.conf_thres,
        #             nms_thres=opt.nms_thres,
        #             img_size=opt.img_size,
        #             batch_size=1,
        #         )

        #         output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #         f=open(output_file_name,'w')

        #         print("Average Precisions:")
        #         for i, c in enumerate(ap_class):
        #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #         print(f"mAP: {AP.mean()}")
        #         f.write("mAP," + str(AP.mean()) + "\n")
        #         f.close()


        #         #with_Ranger_norm

        #         configuration.fault_injection = True
        #         configuration.ranger_on = True
        #         configuration.normalalize_ranger = True
        #         configuration.clipping = False
        #         configuration.layer_specific = False
        #         configuration.grid_specific = False
        #         configuration.save_faulty_parameters = False
        #         configuration.load_faulty_parameters = True
        #         configuration.exponent_bits_only = False
        #         configuration.multiple_locations = mul_loc
        #         configuration.type_fault_injection = "w"
        #         configuration.which_run = runs
        #         configuration.which_bit = -1
        #         configuration.layer_no = -1

        #         del model
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = Darknet(opt.model_def).to(device)
        #         model.load_weights(opt.weights_path)

        #         precision, recall, AP, f1, ap_class = evaluate(
        #             model,
        #             path=valid_path,
        #             iou_thres=opt.iou_thres,
        #             conf_thres=opt.conf_thres,
        #             nms_thres=opt.nms_thres,
        #             img_size=opt.img_size,
        #             batch_size=1,
        #         )

        #         output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #         f=open(output_file_name,'w')

        #         print("Average Precisions:")
        #         for i, c in enumerate(ap_class):
        #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #         print(f"mAP: {AP.mean()}")
        #         f.write("mAP," + str(AP.mean()) + "\n")
        #         f.close()

        #     #Different Scenario
        #     configuration.scenario = "B" #Bit Specific
        #     total_bits = 32
        #     configuration.fault_rate = [1] #changing this so to get statistics faster

        #     for mul_loc in configuration.fault_rate:

        #         for bit_position in range(total_bits):

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = False
        #             configuration.grid_specific = False
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = bit_position
        #             configuration.layer_no = -1

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = False
        #             configuration.grid_specific = False
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = bit_position
        #             configuration.layer_no = -1

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = False
        #             # configuration.grid_specific = False
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "n"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = bit_position
        #             # configuration.layer_no = -1

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()

        #             #Repeat Same procedure for weight

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = False
        #             configuration.grid_specific = False
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = bit_position
        #             configuration.layer_no = -1

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = False
        #             configuration.grid_specific = False
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = bit_position
        #             configuration.layer_no = -1

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = False
        #             # configuration.grid_specific = False
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "w"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = bit_position
        #             # configuration.layer_no = -1

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()


        #     #Different Scenario
        #     # configuration.scenario = "L" #Layer Specific
        #     # total_layers = 75

        #     # for mul_loc in configuration.fault_rate:

        #     #     for layer_n in range(total_layers):

        #     #         configuration.fault_injection = True
        #     #         configuration.ranger_on = False
        #     #         configuration.normalalize_ranger = False
        #     #         configuration.clipping = False
        #     #         configuration.layer_specific = True
        #     #         configuration.grid_specific = False
        #     #         configuration.save_faulty_parameters = True
        #     #         configuration.load_faulty_parameters = False
        #     #         configuration.exponent_bits_only = False
        #     #         configuration.multiple_locations = mul_loc
        #     #         configuration.type_fault_injection = "n"
        #     #         configuration.which_run = runs
        #     #         configuration.which_bit = -1
        #     #         configuration.layer_no = layer_n

        #     #         del model
        #     #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     #         model = Darknet(opt.model_def).to(device)
        #     #         model.load_weights(opt.weights_path)

        #     #         precision, recall, AP, f1, ap_class = evaluate(
        #     #             model,
        #     #             path=valid_path,
        #     #             iou_thres=opt.iou_thres,
        #     #             conf_thres=opt.conf_thres,
        #     #             nms_thres=opt.nms_thres,
        #     #             img_size=opt.img_size,
        #     #             batch_size=1,
        #     #         )

        #     #         output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #     #         f=open(output_file_name,'w')

        #     #         print("Average Precisions:")
        #     #         for i, c in enumerate(ap_class):
        #     #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #     #         print(f"mAP: {AP.mean()}")
        #     #         f.write("mAP," + str(AP.mean()) + "\n")
        #     #         f.close()

        #     #         #with_Ranger

        #     #         configuration.fault_injection = True
        #     #         configuration.ranger_on = True
        #     #         configuration.normalalize_ranger = False
        #     #         configuration.clipping = False
        #     #         configuration.layer_specific = True
        #     #         configuration.grid_specific = False
        #     #         configuration.save_faulty_parameters = False
        #     #         configuration.load_faulty_parameters = True
        #     #         configuration.exponent_bits_only = False
        #     #         configuration.multiple_locations = mul_loc
        #     #         configuration.type_fault_injection = "n"
        #     #         configuration.which_run = runs
        #     #         configuration.which_bit = -1
        #     #         configuration.layer_no = layer_n

        #     #         del model
        #     #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     #         model = Darknet(opt.model_def).to(device)
        #     #         model.load_weights(opt.weights_path)

        #     #         precision, recall, AP, f1, ap_class = evaluate(
        #     #             model,
        #     #             path=valid_path,
        #     #             iou_thres=opt.iou_thres,
        #     #             conf_thres=opt.conf_thres,
        #     #             nms_thres=opt.nms_thres,
        #     #             img_size=opt.img_size,
        #     #             batch_size=1,
        #     #         )

        #     #         output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #     #         f=open(output_file_name,'w')

        #     #         print("Average Precisions:")
        #     #         for i, c in enumerate(ap_class):
        #     #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #     #         print(f"mAP: {AP.mean()}")
        #     #         f.write("mAP," + str(AP.mean()) + "\n")
        #     #         f.close()


        #     #         #with_Ranger_norm

        #     #         configuration.fault_injection = True
        #     #         configuration.ranger_on = True
        #     #         configuration.normalalize_ranger = True
        #     #         configuration.clipping = False
        #     #         configuration.layer_specific = True
        #     #         configuration.grid_specific = False
        #     #         configuration.save_faulty_parameters = False
        #     #         configuration.load_faulty_parameters = True
        #     #         configuration.exponent_bits_only = False
        #     #         configuration.multiple_locations = mul_loc
        #     #         configuration.type_fault_injection = "n"
        #     #         configuration.which_run = runs
        #     #         configuration.which_bit = -1
        #     #         configuration.layer_no = layer_n

        #     #         del model
        #     #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     #         model = Darknet(opt.model_def).to(device)
        #     #         model.load_weights(opt.weights_path)

        #     #         precision, recall, AP, f1, ap_class = evaluate(
        #     #             model,
        #     #             path=valid_path,
        #     #             iou_thres=opt.iou_thres,
        #     #             conf_thres=opt.conf_thres,
        #     #             nms_thres=opt.nms_thres,
        #     #             img_size=opt.img_size,
        #     #             batch_size=1,
        #     #         )

        #     #         output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #     #         f=open(output_file_name,'w')

        #     #         print("Average Precisions:")
        #     #         for i, c in enumerate(ap_class):
        #     #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #     #         print(f"mAP: {AP.mean()}")
        #     #         f.write("mAP," + str(AP.mean()) + "\n")
        #     #         f.close()

        #     #         #Repeat Same procedure for weight

        #     #         configuration.fault_injection = True
        #     #         configuration.ranger_on = False
        #     #         configuration.normalalize_ranger = False
        #     #         configuration.clipping = False
        #     #         configuration.layer_specific = True
        #     #         configuration.grid_specific = False
        #     #         configuration.save_faulty_parameters = True
        #     #         configuration.load_faulty_parameters = False
        #     #         configuration.exponent_bits_only = False
        #     #         configuration.multiple_locations = mul_loc
        #     #         configuration.type_fault_injection = "w"
        #     #         configuration.which_run = runs
        #     #         configuration.which_bit = -1
        #     #         configuration.layer_no = layer_n

        #     #         del model
        #     #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     #         model = Darknet(opt.model_def).to(device)
        #     #         model.load_weights(opt.weights_path)

        #     #         precision, recall, AP, f1, ap_class = evaluate(
        #     #             model,
        #     #             path=valid_path,
        #     #             iou_thres=opt.iou_thres,
        #     #             conf_thres=opt.conf_thres,
        #     #             nms_thres=opt.nms_thres,
        #     #             img_size=opt.img_size,
        #     #             batch_size=1,
        #     #         )

        #     #         output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #     #         f=open(output_file_name,'w')

        #     #         print("Average Precisions:")
        #     #         for i, c in enumerate(ap_class):
        #     #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #     #         print(f"mAP: {AP.mean()}")
        #     #         f.write("mAP," + str(AP.mean()) + "\n")
        #     #         f.close()

        #     #         #with_Ranger

        #     #         configuration.fault_injection = True
        #     #         configuration.ranger_on = True
        #     #         configuration.normalalize_ranger = False
        #     #         configuration.clipping = False
        #     #         configuration.layer_specific = True
        #     #         configuration.grid_specific = False
        #     #         configuration.save_faulty_parameters = False
        #     #         configuration.load_faulty_parameters = True
        #     #         configuration.exponent_bits_only = False
        #     #         configuration.multiple_locations = mul_loc
        #     #         configuration.type_fault_injection = "w"
        #     #         configuration.which_run = runs
        #     #         configuration.which_bit = -1
        #     #         configuration.layer_no = layer_n

        #     #         del model
        #     #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     #         model = Darknet(opt.model_def).to(device)
        #     #         model.load_weights(opt.weights_path)

        #     #         precision, recall, AP, f1, ap_class = evaluate(
        #     #             model,
        #     #             path=valid_path,
        #     #             iou_thres=opt.iou_thres,
        #     #             conf_thres=opt.conf_thres,
        #     #             nms_thres=opt.nms_thres,
        #     #             img_size=opt.img_size,
        #     #             batch_size=1,
        #     #         )

        #     #         output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #     #         f=open(output_file_name,'w')

        #     #         print("Average Precisions:")
        #     #         for i, c in enumerate(ap_class):
        #     #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #     #         print(f"mAP: {AP.mean()}")
        #     #         f.write("mAP," + str(AP.mean()) + "\n")
        #     #         f.close()


        #     #         #with_Ranger_norm

        #     #         configuration.fault_injection = True
        #     #         configuration.ranger_on = True
        #     #         configuration.normalalize_ranger = True
        #     #         configuration.clipping = False
        #     #         configuration.layer_specific = True
        #     #         configuration.grid_specific = False
        #     #         configuration.save_faulty_parameters = False
        #     #         configuration.load_faulty_parameters = True
        #     #         configuration.exponent_bits_only = False
        #     #         configuration.multiple_locations = mul_loc
        #     #         configuration.type_fault_injection = "w"
        #     #         configuration.which_run = runs
        #     #         configuration.which_bit = -1
        #     #         configuration.layer_no = layer_n

        #     #         del model
        #     #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     #         model = Darknet(opt.model_def).to(device)
        #     #         model.load_weights(opt.weights_path)

        #     #         precision, recall, AP, f1, ap_class = evaluate(
        #     #             model,
        #     #             path=valid_path,
        #     #             iou_thres=opt.iou_thres,
        #     #             conf_thres=opt.conf_thres,
        #     #             nms_thres=opt.nms_thres,
        #     #             img_size=opt.img_size,
        #     #             batch_size=1,
        #     #         )

        #     #         output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #     #         f=open(output_file_name,'w')

        #     #         print("Average Precisions:")
        #     #         for i, c in enumerate(ap_class):
        #     #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #     #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #     #         print(f"mAP: {AP.mean()}")
        #     #         f.write("mAP," + str(AP.mean()) + "\n")
        #     #         f.close()

        #     #Different Scenario
        #     configuration.scenario = "G25" #Grid Specific
        #     total_layers = 75
        #     total_grid = 25

        #     configuration.fault_rate = [1]

        #     for mul_loc in configuration.fault_rate:

        #         for grid in range(total_grid):

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = True
        #             # configuration.grid_specific = True
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "n"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = -1
        #             # configuration.layer_no = -1
        #             # configuration.grid_no = grid
        #             # configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             # configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()

        #             #Repeat Same procedure for weight

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = True
        #             # configuration.grid_specific = True
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "w"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = -1
        #             # configuration.layer_no = -1
        #             # configuration.grid_no = grid
        #             # configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             # configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()


        #     #Different Scenario
        #     configuration.scenario = "G15" #Layer Specific
        #     total_layers = 75
        #     total_grid = 15

        #     configuration.fault_rate = [1]

        #     for mul_loc in configuration.fault_rate:

        #         for grid in range(total_grid):

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = True
        #             # configuration.grid_specific = True
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "n"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = -1
        #             # configuration.layer_no = -1
        #             # configuration.grid_no = grid
        #             # configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             # configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()

        #             #Repeat Same procedure for weight

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid
        #             configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = True
        #             # configuration.grid_specific = True
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "w"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = -1
        #             # configuration.layer_no = -1
        #             # configuration.grid_no = grid
        #             # configuration.lower_grid = int(total_layers/total_grid) * (grid)
        #             # configuration.upper_grid = int(total_layers/total_grid) * (grid+1)

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()

        #     #Different Scenario
        #     configuration.scenario = "G1" #Grid Specific (classification layers vs detection layers)
        #     total_layers = 75
        #     total_grid = 2

        #     configuration.fault_rate = [1]

        #     for mul_loc in configuration.fault_rate:

        #         for grid in range(total_grid):

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid

        #             if grid == 0:
        #                 configuration.lower_grid = 0
        #                 configuration.upper_grid = 53
        #             else:
        #                 configuration.lower_grid = 53
        #                 configuration.upper_grid = 75

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "n"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid

        #             if grid == 0:
        #                 configuration.lower_grid = 0
        #                 configuration.upper_grid = 53
        #             else:
        #                 configuration.lower_grid = 53
        #                 configuration.upper_grid = 75

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = True
        #             # configuration.grid_specific = True
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "n"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = -1
        #             # configuration.layer_no = -1
        #             # configuration.grid_no = grid

        #             # if grid == 0:
        #             #     configuration.lower_grid = 0
        #             #     configuration.upper_grid = 53
        #             # else:
        #             #     configuration.lower_grid = 53
        #             #     configuration.upper_grid = 75

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()

        #             #Repeat Same procedure for weight

        #             configuration.fault_injection = True
        #             configuration.ranger_on = False
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = True
        #             configuration.load_faulty_parameters = False
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid

        #             if grid == 0:
        #                 configuration.lower_grid = 0
        #                 configuration.upper_grid = 53
        #             else:
        #                 configuration.lower_grid = 53
        #                 configuration.upper_grid = 75

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()

        #             #with_Ranger

        #             configuration.fault_injection = True
        #             configuration.ranger_on = True
        #             configuration.normalalize_ranger = False
        #             configuration.clipping = False
        #             configuration.layer_specific = True
        #             configuration.grid_specific = True
        #             configuration.save_faulty_parameters = False
        #             configuration.load_faulty_parameters = True
        #             configuration.exponent_bits_only = False
        #             configuration.multiple_locations = mul_loc
        #             configuration.type_fault_injection = "w"
        #             configuration.which_run = runs
        #             configuration.which_bit = -1
        #             configuration.layer_no = -1
        #             configuration.grid_no = grid

        #             if grid == 0:
        #                 configuration.lower_grid = 0
        #                 configuration.upper_grid = 53
        #             else:
        #                 configuration.lower_grid = 53
        #                 configuration.upper_grid = 75

        #             del model
        #             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             model = Darknet(opt.model_def).to(device)
        #             model.load_weights(opt.weights_path)

        #             precision, recall, AP, f1, ap_class = evaluate(
        #                 model,
        #                 path=valid_path,
        #                 iou_thres=opt.iou_thres,
        #                 conf_thres=opt.conf_thres,
        #                 nms_thres=opt.nms_thres,
        #                 img_size=opt.img_size,
        #                 batch_size=1,
        #             )

        #             output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             f=open(output_file_name,'w')

        #             print("Average Precisions:")
        #             for i, c in enumerate(ap_class):
        #                 print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #                 f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             print(f"mAP: {AP.mean()}")
        #             f.write("mAP," + str(AP.mean()) + "\n")
        #             f.close()


        #             #with_Ranger_norm

        #             # configuration.fault_injection = True
        #             # configuration.ranger_on = True
        #             # configuration.normalalize_ranger = True
        #             # configuration.clipping = False
        #             # configuration.layer_specific = True
        #             # configuration.grid_specific = True
        #             # configuration.save_faulty_parameters = False
        #             # configuration.load_faulty_parameters = True
        #             # configuration.exponent_bits_only = False
        #             # configuration.multiple_locations = mul_loc
        #             # configuration.type_fault_injection = "w"
        #             # configuration.which_run = runs
        #             # configuration.which_bit = -1
        #             # configuration.layer_no = -1
        #             # configuration.grid_no = grid

        #             # if grid == 0:
        #             #     configuration.lower_grid = 0
        #             #     configuration.upper_grid = 53
        #             # else:
        #             #     configuration.lower_grid = 53
        #             #     configuration.upper_grid = 75

        #             # del model
        #             # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #             # model = Darknet(opt.model_def).to(device)
        #             # model.load_weights(opt.weights_path)

        #             # precision, recall, AP, f1, ap_class = evaluate(
        #             #     model,
        #             #     path=valid_path,
        #             #     iou_thres=opt.iou_thres,
        #             #     conf_thres=opt.conf_thres,
        #             #     nms_thres=opt.nms_thres,
        #             #     img_size=opt.img_size,
        #             #     batch_size=1,
        #             # )

        #             # output_file_name = "output//results//faulty_gpu_ranger_norm_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #             # f=open(output_file_name,'w')

        #             # print("Average Precisions:")
        #             # for i, c in enumerate(ap_class):
        #             #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #             # print(f"mAP: {AP.mean()}")
        #             # f.write("mAP," + str(AP.mean()) + "\n")
        #             # f.close()


        #Different Scenario
        # configuration.scenario = "FW" #Fix weight fault
        # configuration.fault_rate = [1] #changing this so to get statistics faster

        # for runs in range(configuration.num_runs_FW):

        #     for mul_loc in configuration.fault_rate:

        #         configuration.fault_injection = True
        #         configuration.ranger_on = False
        #         configuration.normalalize_ranger = False
        #         configuration.clipping = False
        #         configuration.layer_specific = False
        #         configuration.grid_specific = False
        #         configuration.save_faulty_parameters = True
        #         configuration.load_faulty_parameters = False
        #         configuration.exponent_bits_only = False
        #         configuration.multiple_locations = mul_loc
        #         configuration.type_fault_injection = "w"
        #         configuration.which_run = runs
        #         configuration.which_bit = -1
        #         configuration.layer_no = -1

        #         # del model
        #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = Darknet(opt.model_def).to(device)
        #         model.load_weights(opt.weights_path)

        #         precision, recall, AP, f1, ap_class = evaluate(
        #             model,
        #             path=valid_path,
        #             iou_thres=opt.iou_thres,
        #             conf_thres=opt.conf_thres,
        #             nms_thres=opt.nms_thres,
        #             img_size=opt.img_size,
        #             batch_size=1,
        #         )

        #         output_file_name = "output//results//faulty_gpu_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

        #         f=open(output_file_name,'w')

        #         print("Average Precisions:")
        #         for i, c in enumerate(ap_class):
        #             print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        #             f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

        #         print(f"mAP: {AP.mean()}")
        #         f.write("mAP," + str(AP.mean()) + "\n")
        #         f.close()

                #with_Ranger

                # configuration.fault_injection = True
                # configuration.ranger_on = True
                # configuration.normalalize_ranger = False
                # configuration.clipping = False
                # configuration.layer_specific = False
                # configuration.grid_specific = False
                # configuration.save_faulty_parameters = False
                # configuration.load_faulty_parameters = True
                # configuration.exponent_bits_only = False
                # configuration.multiple_locations = mul_loc
                # configuration.type_fault_injection = "w"
                # configuration.which_run = runs
                # configuration.which_bit = -1
                # configuration.layer_no = -1

                # del model
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # model = Darknet(opt.model_def).to(device)
                # model.load_weights(opt.weights_path)

                # precision, recall, AP, f1, ap_class = evaluate(
                #     model,
                #     path=valid_path,
                #     iou_thres=opt.iou_thres,
                #     conf_thres=opt.conf_thres,
                #     nms_thres=opt.nms_thres,
                #     img_size=opt.img_size,
                #     batch_size=1,
                # )

                # output_file_name = "output//results//faulty_gpu_ranger_" + str(configuration.scenario) + "_type_" + str(configuration.type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(configuration.which_bit) + "_layer_" + str(configuration.layer_no) + "_grid_" + str(configuration.grid_no) + "_nr_" + str(runs) + ".txt"

                # f=open(output_file_name,'w')

                # print("Average Precisions:")
                # for i, c in enumerate(ap_class):
                #     print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
                #     f.write(str(class_names[c]) + "," + str(AP[i]) + "\n")

                # print(f"mAP: {AP.mean()}")
                # f.write("mAP," + str(AP.mean()) + "\n")
                # f.close()
