import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import tensorflow as tf
from train_detector import cnn_model_struct
import tqdm

class Tester:
    def __init__(self,config):
        self.config = config
        self.input = tf.placeholder(tf.float32,
                            [None,config.image_target_size[0],config.image_target_size[1],config.image_target_size[2]], name='ip_placeholder')
        self.initialized = False

        with tf.device('/gpu:0'):
            with tf.variable_scope("model") as scope:
                self.model = cnn_model_struct()
                self.model.build(self.input, config.num_classes, train_mode=False)

            self.gpuconfig = tf.ConfigProto()
            self.gpuconfig.gpu_options.allow_growth = True
            self.gpuconfig.allow_soft_placement = True
            self.saver = tf.train.Saver()

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self,item)

    def make_predictions(self, patches):
        try:
            probs = []
            if self.initialized == False:
                self.sess = tf.Session(config=self.gpuconfig)
                ckpts = tf.train.latest_checkpoint(self.config.model_output)
                self.saver.restore(self.sess, ckpts)
                self.initialized = True
            probs = self.sess.run(self.model.output,feed_dict={self.input:patches})

            # with tf.Session(config=self.gpuconfig) as sess:
            #     if self.initialized == False:
            #         ckpts = tf.train.latest_checkpoint(self.config.model_output)
            #         self.saver.restore(sess, ckpts)
            #         self.initialized = True
            #     probs = sess.run(self.model.output,feed_dict={self.input:patches})
        except tf.errors.NotFoundError:
            print ('checkpoint could not be restored')
        finally:
            return probs

"""
Extract image patches and put them through the model 
"""
def get_and_test_patches(
        rgb_frame,
        config,
):
    tester = Tester(config=config)
    patches = []
    sz = rgb_frame.shape

    half_win_width, half_win_height = config.image_target_size[0]/2, config.image_target_size[1]/2
    counter = 0
    #start_x, end_x = 700, 730
    #start_y, end_y = 126, 156
    start_x, end_x = 700, 1050
    start_y, end_y = 126, 522
    # start_x, end_x = half_win_width+1, sz[0]-half_win_width-1
    # start_y, end_y = half_win_height+1,sz[1]-half_win_height-1
    results = []
    for x in tqdm.tqdm(range(start_x, end_x)):
        for y in range(start_y, end_y):
            patch = rgb_frame[(x - half_win_width - 1):(x + half_win_width - 1),
                    (y - half_win_height - 1):(y + half_win_height - 1), :]
            patches.append(patch.astype(np.float32))
            counter+=1
            #plt.imshow(patch)
            #plt.show()
            if counter%config.test_batch == 0:
                probs = tester.make_predictions(patches=patches)
                results.append(probs)
                patches= []
    if patches != []:
        probs = tester.make_predictions(patches=patches)
        results.append(probs)

    import ipdb; ipdb.set_trace()
    vis_res = np.reshape(np.concatenate(results,axis=0)[:,1],[end_x-start_x,end_y-start_y])
    plt.subplot(121)
    plt.imshow(vis_res)
    plt.subplot(122)
    plt.imshow(rgb_frame[start_x:end_x,start_y:end_y,:])
    plt.show()
    print ('end of function')


def eval_video_with_model(config=None):
    # Load in the video to read
    video_stream = cv2.VideoCapture(os.path.join(config.base_dir, config.video_name))
    frameid = 0
    max_lim = 5000
    bar = progressbar.ProgressBar(max_value=max_lim)

    print 'Reading from video...'
    while (video_stream.isOpened()):
        flag, frame = video_stream.read()
        assert flag, 'Reading from video has failed!'

        # get the current frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # make patches from the frame and test
        pred_frame = get_and_test_patches(rgb_frame=rgb_frame,
                                             config=config)
        plt.imshow(pred_frame)
        plt.show()

        frameid = frameid + 1
        bar.update(frameid)
        if frameid == max_lim:
            break

    print 'Finished...'

    # # import ipdb; ipdb.set_trace();
    # total_items = len(all_patches)
    # arr = np.arange(total_items)
    # np.random.shuffle(arr)
    #
    # all_patches = np.asarray(all_patches)[arr]
    # all_labels = np.asarray(all_labels)[arr]
    # #assert (np.sum(data_prop.values()) != 1.), 'Train vs Test split specified incorrectly'
    #
    # train_idx_lim = int(data_prop['train'] * total_items)
    # val_idx_lim = int((data_prop['train'] + data_prop['val']) * total_items)
    #
    # # write the tf records as specified
    # if config.train_tfrecords != None:
    #     write_tfrecord(
    #         tfrecord_dir,
    #         config.train_tfrecords,
    #         all_patches[:train_idx_lim],
    #         all_labels[:train_idx_lim])
    #
    # if config.val_tfrecords != None:
    #     write_tfrecord(
    #         tfrecord_dir,
    #         config.val_tfrecords,
    #         all_patches[train_idx_lim:val_idx_lim],
    #         all_labels[train_idx_lim:val_idx_lim])
    #
    # if config.test_tfrecords != None:
    #     write_tfrecord(
    #         tfrecord_dir,
    #         config.test_tfrecords,
    #         all_patches[val_idx_lim:],
    #         all_labels[val_idx_lim:])