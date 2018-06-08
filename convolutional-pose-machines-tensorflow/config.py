class FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 256
    image_size = 256 # input_size = 256
    heatmap_size = 32
    hmap_size = 32 # heatmap_size = 32
    cpm_stages = 3
    total_stages = 3 # cpm_stages = 6
    joint_gaussian_variance = 1.0
    center_radius = 21
    num_of_joints = 21
    total_joints = 21 # num_of_joints = 21
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0
    box_size = 64
    optimizer = 'RMSProp'

    """
    Demo settings
    """
    # 'MULTI': show multiple stage heatmaps
    # 'SINGLE': show last stage heatmap
    # 'Joint_HM': show last stage heatmap for each joint
    # 'image or video path': show detection on single image or video
    DEMO_TYPE = '/home/qiaohe/qiao_img' #'SINGLE'

    #model_path =  'cpm_hand-25000'
    cam_id = 0

    webcam_height = 480
    webcam_width = 640

#     use_kalman = False
#     kalman_noise = 0.03


    """
    Training settings
    """
    network_def = 'cpm_model'
    train_tf_file = 'train.tfrecords' #'train.tfrecords'#
    val_tf_file = 'test.tfrecords' #'test.tfrecords'#
    bg_img_dir = ''
    pretrained_model = 'cpm_model-10000'#'cpm_model-32000'
    batch_size = 5
    init_lr = 0.001#0.005
    lr = 0.001#0.001 #0.0005#
    lr_decay_rate = 0.5
    l_decay_rate = 0.5
    lr_decay_step = 10000
    l_decay_step = 10000
    training_iters = 10000 #300000
    verbose_iters = 10
    validation_iters = 500 #50 #1000 
    model_save_iters = 2500 #100 #5000 
    intermediate_showing = 1000
    validation_batch_per_iter = 20
    padding = 30
    augmentation_config = {'hue_shift_limit': (-5, 5),
                           'sat_shift_limit': (-10, 10),
                           'val_shift_limit': (-15, 15),
                           'translation_limit': (-0.15, 0.15),
                           'scale_limit': (-0.3, 0.5),
                           'rotate_limit': (-90, 90)}
    hnm = True  # Make sure generate hnm files first
    do_cropping = True

    """
    For Freeze graphs
    """
    output_node_names = 'stage_3/Mconv7/BiasAdd:0' #'Stage_3/Mconv7/BiasAdd:0'


    """
    For Drawing
    """
    # Default Pose
    default_hand = [[259, 335],
                    [245, 311],
                    [226, 288],
                    [206, 270],
                    [195, 261],
                    [203, 308],
                    [165, 290],
                    [139, 287],
                    [119, 284],
                    [199, 328],
                    [156, 318],
                    [128, 314],
                    [104, 318],
                    [204, 341],
                    [163, 340],
                    [133, 347],
                    [108, 349],
                    [206, 359],
                    [176, 368],
                    [164, 370],
                    [144, 377]]

    limbs = [[0, 1],[1 , 2],[2 , 3],[3 , 4],
                [0, 5],[5 , 6],[6 , 7],[7 , 8],
                [0, 9],[9 ,10],[10,11],[11,12],
                [0,13],[13,14],[14,15],[15,16],
                [0,17],[17,18],[18,19],[19,20]]

    # Finger colors
    joint_color_code = [
        [100.,  100.,  100.], 
        [100.,    0.,    0.],
        [150.,    0.,    0.],
        [200.,    0.,    0.],
        [255.,    0.,    0.],
        [100.,  100.,    0.],
        [150.,  150.,    0.],
        [200.,  200.,    0.],
        [255.,  255.,    0.],
        [  0.,  100.,   50.],
        [  0.,  150.,   75.],
        [  0.,  200.,  100.],
        [  0.,  255.,  125.],
        [  0.,   50.,  100.],
        [  0.,   75.,  150.],
        [  0.,  100.,  200.],
        [  0.,  125.,  255.],
        [100.,    0.,  100.],
        [150.,    0.,  150.],
        [200.,    0.,  200.],
        [255.,    0.,  255.]]











