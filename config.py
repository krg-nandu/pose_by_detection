class Config(object):

    def __init__(self):
        #data directories
        self.base_dir = '/media/data_cifs/ruth/6October17'
        self.video_name = 'context.avi'
        self.label_dir = '/media/data_cifs/lakshmi/zebrafish/groundtruths/'
        self.tfrecord_dir = '/media/data_cifs/lakshmi/tfrecords/'

        self.objects_to_include = [0,1,2,3,4,5] #fish ids to use!
        self.joints_to_extract = [0]
        self.data_prop = {'train':0.75,'val':0.1,'test':0.15}

        #tfrecords configuration
        self.train_tfrecords = 'train.tfrecords'
        self.val_tfrecords = 'val.tfrecords'
        self.test_tfrecords = 'test.tfrecords'

        self.results_dir = ''
        self.model_output = ''
        self.model_input = ''
        self.train_summaries = ''

        # self.vgg16_weight_path = pjoin(
        #     '/media/data_cifs/clicktionary/',
        #     'pretrained_weights',
        #     'vgg16.npy')

        #model settings
        self.model_type = 'vgg_regression_model_4fc'
        self.epochs = 100
        self.image_target_size = [28,28,3]
        self.label_shape = 36
        self.train_batch = 32
        self.val_batch= 32
        #self.initialize_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        #self.fine_tune_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        #self.batch_norm = ['conv1','fc1','fc2']
        #self.wd_layers = ['fc6', 'fc7', 'pre_fc8']
        #self.wd_penalty = 0.005
        #self.optimizier = 'adam'
        #self.lr = 1e-4  # Tune this -- also try SGD instead of ADAm
        #self.hold_lr = self.lr / 2
        #self.keep_checkpoints = 100

        #training setting
        self.num_classes = 2