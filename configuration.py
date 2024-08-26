
#batch is set to 1 always for easier debugging

def init():

    global num_runs #number of runs
    global which_run
    global fault_rate
    global fault_injection
    global type_fault_injection
    global layer_specific #random fault inside one specific layer
    global layer_no
    global total_conv_layers
    global exponent_bits_only
    global bits_specific
    global which_bit
    global save_faulty_parameters
    global load_faulty_parameters
    # global random_neuron
    # global random_weight
    global dataset
    global percent_images
    global num_images
    global save_files_ranger
    global ranger_on
    global file_name_max
    global file_name_min
    global execute_everything
    global multiple_locations
    global normalalize_ranger # Florian suggestion
    global clipping # Florian suggestion
    global scenarios
    global scenario
    global grid_specific
    global grid_no
    global lower_grid
    global upper_grid
    global num_runs_FW #no. of runs of fixed weight faults

    scenarios = ["R, B, L, G25, G15, G1, FW"] #Random, Bit-specific, Layer-specific, Grid Specific, FW
    scenario = "R"
    num_runs_FW = 500

    grid_specific = False
    grid_no = -1
    lower_grid = -1
    upper_grid = -1

    exponent_bits_only = True
    bits_specific = False
    which_bit = 1 #select from 0 to 31 (-1 then randomly select from 0 to 31)
    num_runs = 10 #number of runs
    which_run = 0
    multiple_locations = 1 #just a flag, set it to >=2 for multiple locations
    fault_rate = [1]#, 2, 3, 4, 5, 10, 15, 20] #, 40, 50]
    execute_everything = True
    save_faulty_parameters = True
    load_faulty_parameters = False
    fault_injection = True
    type_fault_injection = "w" #weight or neuron
    layer_specific = False
    total_conv_layers = 75
    layer_no = 1
    # random_neuron = False
    # random_weight = False
    dataset = "valid" #"train" #"valid"
    percent_images = 100 #percentage of images if 0 -> then number of images will be used
    num_images = 10
    save_files_ranger = False
    ranger_on = False
    normalalize_ranger = False
    clipping = False
    file_name_max = "//mnt//raid0//trajectorypred//Datasets//COCO//ranger_files//tensor_max_10.pt"
    file_name_min = "//mnt//raid0//trajectorypred//Datasets//COCO//ranger_files//tensor_min_10.pt"

