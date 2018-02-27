from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import tensorflow as tf
import config
from tf_data_handler import inputs

class cnn_model_struct:
    def __init__(self, trainable=False):
        self.trainable = trainable
        self.data_dict = None
        self.var_dict = {}

    def __getitem__(self, item):
        return getattr(self,item)

    def __contains__(self, item):
        return hasattr(self,item)

    def build(self, patch, output_shape,train_mode=None):
        print ("building the network")
        input_patch = tf.identity(patch,name="input_patch")
        with tf.name_scope('reshape'):
            x_image = tf.reshape(input_patch, [-1, 28, 28, 3])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_variable([5, 5, 3, 32],var_name='wconv1')
            self.b_conv1 = self.bias_variable([32],var_name='bconv1')
            self.h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_variable([5, 5, 32, 64],var_name='wconv2')
            self.b_conv2 = self.bias_variable([64],var_name='bconv2')
            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024],var_name='wfc1')
            self.b_fc1 = self.bias_variable([1024],var_name='bfc1')

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            if train_mode == True:
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, 0.7)
            else:
                self.h_fc1_drop = tf.nn.dropout(self.h_fc1, 1.0)

        # Map the 1024 features to "numclass" classes, one for each digit
        with tf.name_scope('fc2'):
            self.W_fc2 = self.weight_variable([1024, output_shape],var_name='wfc2')
            self.b_fc2 = self.bias_variable([output_shape],var_name='bfc2')

            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
            self.y_conv = tf.nn.sigmoid(self.y_conv)
        #return y_conv, keep_prob
        #return y_conv
        self.output = tf.identity(self.y_conv,name="output")

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape, var_name):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.get_variable(name=var_name,initializer=initial)

    def bias_variable(self, shape, var_name):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.001, shape=shape)
        return tf.get_variable(name=var_name,initializer=initial)

def train_model(config):
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    val_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)

    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(tfrecord_file=train_data,
                                            num_epochs=config.epochs,
                                            image_target_size=config.image_target_size,
                                            label_shape=config.num_classes,
                                            batch_size=config.train_batch,
                                            augmentation=True)

        val_images, val_labels = inputs(tfrecord_file=val_data,
                                        num_epochs=config.epochs,
                                        image_target_size=config.image_target_size,
                                        label_shape=config.num_classes,
                                        batch_size=config.val_batch)

    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            print ("creating the model")
            # Create the model
            # x = tf.placeholder(tf.float32, [None, config.image_target_size[0],config.image_target_size[1],config.image_target_size[2]])
            # y_ = tf.placeholder(tf.int64, [None,1])

            # Build the graph for the deep net
            #y_conv, keep_prob = deepnn(train_images)
            #y_conv = deepnn(train_images)
            model = cnn_model_struct()
            model.build(train_images,config.num_classes,train_mode=True)
            y_conv = model.output

            y_ = tf.cast(train_labels,tf.int64)
            yhat = tf.argmax(y_conv,1)

            # Define loss and optimizer
            with tf.name_scope('loss'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(
                    labels=y_, logits=y_conv)
            cross_entropy = tf.reduce_mean(cross_entropy)

            with tf.name_scope('adam_optimizer'):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
                correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)

            print("using validation")
            scope.reuse_variables()
            val_model = cnn_model_struct()
            val_model.build(val_images, config.num_classes, train_mode=False)
            val_results = tf.argmax(val_model.output, 1)
            val_error = tf.reduce_mean(tf.cast(tf.equal(val_results, tf.cast(val_labels, tf.int64)), tf.float32))

            tf.summary.scalar("loss", cross_entropy)
            tf.summary.scalar("train error", accuracy)
            tf.summary.scalar("validation error", val_error)
            summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(tf.global_variables())

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement = True

    with tf.Session(config=gpuconfig) as sess:
        graph_location = tempfile.mkdtemp()
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        #import ipdb; ipdb.set_trace();
        #batch_images, batch_labels = sess.run([train_images, train_labels])
        #val_batch_images, val_batch_labels = sess.run([val_images, val_labels])

        step = 0
        try:
            while not coord.should_stop():
                #train for a step
                _, tr_images, tr_labels, loss, softmax_outputs, pred_labels, error = sess.run([train_step,train_images,train_labels, cross_entropy, y_conv, yhat, accuracy])
                print("step={}, loss={}, accuracy={}".format(step,loss,error))
                step+=1
                #validate model
                if step % 200 == 0:
                    vl_img, vl_lab, vl_res, vl_err = sess.run([val_images,val_labels,val_results,val_error])
                    print("\t val error = {}".format(vl_err))
                    #import ipdb; ipdb.set_trace();
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str,step)
                # save the model check point
                if step % 1000 == 0:
                    saver.save(sess,os.path.join(
                        config.model_output,
                        config.model_name+'_'+str(step)+'.ckpt'
                    ),global_step=step)

        except tf.errors.OutOfRangeError:
            print("Finished training for %d epochs" % config.epochs)
        finally:
            coord.request_stop()
            coord.join(threads)

def test_model_eval(config):
    test_data = os.path.join(config.tfrecord_dir, config.test_tfrecords)
    with tf.device('/cpu:0'):
        test_images, test_labels = inputs(tfrecord_file=test_data,
                                            num_epochs=None,
                                            image_target_size=config.image_target_size,
                                            label_shape=config.num_classes,
                                            batch_size=config.test_batch,
                                            augmentation=False)

    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            model = cnn_model_struct()
            model.build(test_images,config.num_classes,train_mode=False)
            results = tf.argmax(model.output, 1)
            error = tf.reduce_mean(tf.cast(tf.equal(results, tf.cast(test_labels, tf.int64)), tf.float32))

        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        with tf.Session(config=gpuconfig) as sess:
            #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            #sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            step=0
            try:
                while not coord.should_stop():
                    # load the model here
                    ckpts=tf.train.latest_checkpoint(config.model_output)
                    saver.restore(sess,ckpts)
                    ims, labs, probs, err, res = sess.run([test_images,test_labels,model.output,error,results])
                    import ipdb; ipdb.set_trace();
            except tf.errors.OutOfRangeError:
                print('Epoch limit reached!')
            finally:
                coord.request_stop()
            coord.join(threads)

def get_model_predictions(config,patches):
    input = tf.placeholder(tf.float32, [None,config.image_target_size[0],config.image_target_size[1],config.image_target_size[2]], name='ip_placeholder')
    with tf.device('/gpu:0'):
        with tf.variable_scope("model") as scope:
            model = cnn_model_struct()
            model.build(input,config.num_classes,train_mode=False)

        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True
        gpuconfig.allow_soft_placement = True
        saver = tf.train.Saver()

        with tf.Session(config=gpuconfig) as sess:
            #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            #sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            step=0
            try:
                while not coord.should_stop():
                    # load the model here
                    ckpts=tf.train.latest_checkpoint(config.model_output)
                    saver.restore(sess,ckpts)
                    ims, probs = sess.run(model.output,feed_dict={input:patches})
            except tf.errors.OutOfRangeError:
                print('Epoch limit reached!')
            finally:
                coord.request_stop()
            coord.join(threads)
    return probs