import numpy as np

#Random single bitflip neuron fault injection

# fault_parameters = "output//fault_parameters.txt"
# fault_parameters_layer = "output//fault_parameters_l_" + str(configuration.layer_no) + ".txt"

# result_normal = "output//result_normal.txt"
# result_faulty_single_bit = "output//result_faulty_single_bit.txt"
# result_faulty_single_bit_ranger = "output//result_faulty_single_bit_ranger.txt"
# result_faulty = "output//result_faulty.txt"
# result_faulty_ranger = "output//result_faulty_ranger.txt"

# file_result_normal=open(result_normal,'r')
# file_result_faulty_single_bit=open(result_faulty_single_bit,'r')
# file_result_faulty_single_bit_ranger=open(result_faulty_single_bit_ranger,'r')
# file_result_faulty=open(result_faulty,'r')
# file_result_faulty_ranger=open(result_faulty_ranger,'r')

# lines_file_result_normal = file_result_normal.readlines()
# lines_file_result_faulty_single_bit = file_result_faulty_single_bit.readlines()
# lines_file_result_faulty_single_bit_ranger = file_result_faulty_single_bit_ranger.readlines()
# lines_file_result_faulty = file_result_faulty.readlines()
# lines_file_result_faulty_ranger = file_result_faulty_ranger.readlines()

# file_result_normal.close()
# file_result_faulty_single_bit.close()
# file_result_faulty_single_bit_ranger.close()
# file_result_faulty.close()
# file_result_faulty_ranger.close()


#general:
# plot 1: mAP vs fault-rate (currently i am using 1 fault per image per sec ) -> increase fault rate and see accuracy

# total_runs = 1
# fault_rate = [1, 2, 3, 4, 5, 10, 15, 20] #, 40, 50]
# layer = -1
# grid = -1
# bit = -1
# type_fault_injection = "w"
# no_faults = 1
# which_run = 0
# plot_1 = "output//plots//fault_rates_w_ranger_norm_gpu.txt"
# file_plot_1 = open(plot_1,'w')


# for runs in range(total_runs):
#     for mul_loc in fault_rate:
#         # file_fr = open("output//result_faulty_" + str(type_fault_injection) + "_mul_loc_" + str(mul_loc) + "_nr_" + str(runs) + ".txt", 'r')
#         # file_fr = open("output//results//result_faulty_exp_1_" + str(type_fault_injection) + "_mul_loc_" + str(mul_loc) + "_nr_" + str(runs) + ".txt", 'r')
#         file_fr = open("output//results//faulty_gpu_ranger_norm_R_type_" + str(type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(bit) + "_layer_" + str(layer) + "_grid_" + str(grid) + "_nr_" + str(which_run) + ".txt", 'r')
#         lines_file = file_fr.readlines()
#         mAP_ = float(lines_file[len(lines_file)-1].split(",")[1])
#         file_plot_1.write(str(mul_loc) + "," + str(mAP_) + "\n")
#         file_fr.close()

# file_plot_1.close()

#Fixed Weights:
# plot 1: mAP vs fault-rate (currently i am using 1 fault per image per sec ) -> increase fault rate and see accuracy

total_runs = 50
fault_rate = [1]#, 2, 3, 4, 5, 10, 15, 20] #, 40, 50]
layer = -1
grid = -1
bit = -1
type_fault_injection = "w"
plot_1 = "output//plots//fault_rates_R_w_gpu_runs.txt"
file_plot_1 = open(plot_1,'w')


for runs in range(2, total_runs):
    for mul_loc in fault_rate:
        # file_fr = open("output//result_faulty_" + str(type_fault_injection) + "_mul_loc_" + str(mul_loc) + "_nr_" + str(runs) + ".txt", 'r')
        # file_fr = open("output//results//result_faulty_exp_1_" + str(type_fault_injection) + "_mul_loc_" + str(mul_loc) + "_nr_" + str(runs) + ".txt", 'r')
        file_fr = open("output//results//faulty_gpu_R_type_" + str(type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(bit) + "_layer_" + str(layer) + "_grid_" + str(grid) + "_nr_" + str(runs) + ".txt", 'r')
        lines_file = file_fr.readlines()
        mAP_ = float(lines_file[len(lines_file)-1].split(",")[1])
        file_plot_1.write(str(runs) + "," + str(mAP_) + "\n")
        file_fr.close()

file_plot_1.close()


# #Grid_wise:
# # plot 1: mAP vs Grids (currently i am using 1 fault per image per sec ) -> increase fault rate and see accuracy

# total_runs = 1
# total_grids = 2
# fault_rate = [1]#, 2, 3, 4, 5, 10, 15, 20] #, 40, 50]
# layer = -1
# grid = -1
# bit = -1
# type_fault_injection = "n"
# no_faults = 1
# which_run = 0
# plot_1 = "output//plots//fault_rates_G1_n_ranger_gpu.txt"
# file_plot_1 = open(plot_1,'w')


# for grid in range(total_grids):
#     for mul_loc in fault_rate:
#         # file_fr = open("output//result_faulty_" + str(type_fault_injection) + "_mul_loc_" + str(mul_loc) + "_nr_" + str(runs) + ".txt", 'r')
#         # file_fr = open("output//results//result_faulty_exp_1_" + str(type_fault_injection) + "_mul_loc_" + str(mul_loc) + "_nr_" + str(runs) + ".txt", 'r')
#         file_fr = open("output//results//faulty_gpu_ranger_G1_type_" + str(type_fault_injection) + "_faults_" + str(mul_loc) + "_bit_" + str(bit) + "_layer_" + str(layer) + "_grid_" + str(grid) + "_nr_" + str(which_run) + ".txt", 'r')
#         lines_file = file_fr.readlines()
#         mAP_ = float(lines_file[len(lines_file)-1].split(",")[1])
#         file_plot_1.write(str(grid) + "," + str(mAP_) + "\n")
#         file_fr.close()

# file_plot_1.close()

    


#general 
# plot 2: mAP vs all scenerios (currently only neuron but add weight statistics also) 
#(single bit random neuron, random neuron injection, normal run, with ranger, with ranger & single bit faults, with ranger and random faults)

# list_mAP_names = ["normal", "singlebitflip", "ranger_bitflip", "fault_random_value", "ranger_randomfault"]
# mAP_normal = float(lines_file_result_normal[len(lines_file_result_normal)-1].split(",")[1])
# mAP_single_bit = float(lines_file_result_faulty_single_bit[len(lines_file_result_faulty_single_bit)-1].split(",")[1])
# mAP_single_bit_ranger = float(lines_file_result_faulty_single_bit_ranger[len(lines_file_result_faulty_single_bit_ranger)-1].split(",")[1])
# mAP_faulty = float(lines_file_result_faulty[len(lines_file_result_faulty)-1].split(",")[1])
# mAP_faulty_ranger = float(lines_file_result_faulty_ranger[len(lines_file_result_faulty_ranger)-1].split(",")[1])

# list_mAP_values = [mAP_normal, mAP_single_bit, mAP_single_bit_ranger, mAP_faulty, mAP_faulty_ranger]

# plot_2 = "output//plots//mAPvsall.txt"
# file_plot_2 = open(plot_2,'w')

# for i in range(len(list_mAP_names)):
#     file_plot_2.write(list_mAP_names[i] + "," + str(list_mAP_values[i]) + "\n")

# file_plot_2.close()


#specific: Layer sensitivity 
# plot 3: mAP vs all layers (which layers are more sensitive) 

# total_layers = 75
# type_fault_injection = "w"
# multiple_locations = 0
# which_run = 0
# mAP_layers = []
# plot_3 = "output//plots//mAPvslayers_w.txt"
# file_plot_3 = open(plot_3,'w')

# for i in range(total_layers):
#     # file_layer = open("output//result_faulty_l_"+ str(i) +".txt", 'r')
#     file_layer = open("output//result_faulty_l_" + str(i) + "_" + str(type_fault_injection) + "_mul_loc_" + str(multiple_locations) + "_nr_" + str(which_run) + ".txt", 'r')
#     lines_file = file_layer.readlines()
#     mAP_ = float(lines_file[len(lines_file)-1].split(",")[1])
#     file_plot_3.write("l_" + str(i) + "," + str(mAP_) + "\n")
#     file_layer.close()

# file_plot_3.close()

# #specific: which bit locations are sensitive 
# # plot 4: mAP vs bit location 

# total_bits = 32
# layer = -1
# grid = -1
# type_fault_injection = "n"
# no_faults = 1
# which_run = 0
# mAP_layers = []
# plot_3 = "output//plots//mAPvsbits_n_ranger.txt"
# file_plot_3 = open(plot_3,'w')

# for i in range(total_bits):
#     # file_layer = open("output//result_faulty_l_"+ str(i) +".txt", 'r')
#     # faulty_gpu_B_type_n_faults_1_bit_0_layer_-1_grid_-1_nr_0.txt
#     file_layer = open("output//results//faulty_gpu_ranger_B_type_" + str(type_fault_injection) + "_faults_" + str(no_faults) + "_bit_" + str(i) + "_layer_" + str(layer) + "_grid_" + str(grid) + "_nr_" + str(which_run) + ".txt", 'r')
#     lines_file = file_layer.readlines()
#     mAP_ = float(lines_file[len(lines_file)-1].split(",")[1])
#     file_plot_3.write("b_" + str(i) + "," + str(mAP_) + "\n")
#     file_layer.close()

# file_plot_3.close()


#specific: class sensitivity (1 plot)
# plot 5: mAP vs class (you have calculate mAP for each class on all the scenarios)

# total_layers = 75
# type_fault_injection = "w"
# multiple_locations = 0
# which_run = 0

# class_names = []
# for i in range(len(lines_file_result_normal)):
#     if i == len(lines_file_result_normal)-1:
#         break
#     class_names.append(str(lines_file_result_normal[i].split(",")[0]))

# AP_list = []

# i = 0
# for i in range(total_layers):
#     # file_layer = open("output//result_faulty_l_"+ str(i) +".txt", 'r')
#     file_layer = open("output//result_faulty_l_" + str(i) + "_" + str(type_fault_injection) + "_mul_loc_" + str(multiple_locations) + "_nr_" + str(which_run) + ".txt", 'r')

#     lines_file = file_layer.readlines()
#     file_layer.close()

#     l, k = 0, 0

#     if i == 0:
#         for l in range(len(lines_file)):
#             AP_ = float(lines_file[l].split(",")[1])
#             AP_list.append(AP_)
#     else:
#         for k in range(len(lines_file)):
#             AP_ = float(lines_file[k].split(",")[1])
#             mean_AP = (AP_list[k] + AP_) / 2 
#             AP_list[k] = mean_AP


# plot_5 = "output//plots//mAPvsclass_w.txt"
# file_plot_5 = open(plot_5,'w')
# i=0
# for i in range(len(AP_list)-1):
#     file_plot_5.write(class_names[i] + "," + str(AP_list[i]) + "\n")

# file_plot_5.close()

#TODO

#specific: class-layer sensitivity (for particular class which layer is most sensitive) (75 plots)
# plot 6: AP vs layer-class 


