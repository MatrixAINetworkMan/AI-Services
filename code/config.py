
class Config(object):
    version_id = "02"

    dev_directory       = "./dev_data"
    eval_directory      = "./eval_data"
    # dev_directory       = "./small_dev_data"
    # eval_directory      = "./small_eval_data"
    model_directory     = "./model"
    feature_directory   = "./feature"
    result_directory    = "./result"
    result_file         = "result.csv"

    # device
    cuda = "cuda:0"

    # pAUC
    max_fpr = 0.1

    # audio feature:
    n_fft = 2046
    hop_length = 512

    split_length = 32
    stride = 16

    # ArcFace loss
    s = 30
    m = 0.05

    # train
    num_epochs = 20
    batch_size = 64

    model_save_interval = 5

    # learning rate strategy
    lr = 0.05
    lr_strategy = True

    ## StepLR
    stepLR = False
    step_size = 5
    gamma = 0.5
    ## CosineAnnealingLR
    T_max = 20
    eta_min = 5e-4

    # optimizer
    optimizer = "SGD"
    momentum = 0.9
    nesterov = True
    weight_decay = 0.0005

    num_workers = 0

    # test
    mode = True     # True for dev_data, False for eval_data

