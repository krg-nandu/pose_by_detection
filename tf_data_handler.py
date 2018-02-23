import tensorflow as tf
import tqdm
import numpy as np
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(
        tf_record_dir='',
        tf_filename='',
        images=None,
        labels=None):
    writer = tf.python_io.TFRecordWriter(os.path.join(
        tf_record_dir,
        tf_filename))
    for i in tqdm.tqdm(range(len(images))):
        #feature = {'label': _int64_feature(labels[i]),
        #           'image': _bytes_feature(images[i].tostring())}
        feature = {'label': _bytes_feature(labels[i].tostring()),
                   'image': _bytes_feature(images[i].tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()

def fliplr(x):
    return tf.image.flip_left_right(x)

def flipud(x):
    return tf.image.flip_up_down(x)

def adjust_bright(x):
    return tf.image.random_brightness(x,max_delta=0.5)

def no_aug(x):
    return x

def augment_patch(patch):
    options = {
        0: fliplr,
        1: flipud,
        2: adjust_bright,
        3: no_aug,
    }
    aug_method = np.random.randint(4, size=1)
    return options[aug_method[0]](patch)

# establish the data queue
def inputs(
        tfrecord_file,
        num_epochs,
        image_target_size,
        label_shape,
        batch_size,
        augmentation=False):

    with tf.name_scope('input'):
        if os.path.exists(tfrecord_file) is False:
            print("{} not exists".format(tfrecord_file))
        feature = {
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        }
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=num_epochs)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)
        label = tf.decode_raw(features['label'], tf.float32)

        # Cast label data into int32
        label.set_shape(1)
        label = tf.cast(tf.reshape(label,[]),tf.int64)
        #label = tf.cast(feature['label'],tf.int32)
        '''
        this is for one-hot encoding. better to use it online during training
        '''
        #label = tf.one_hot(tf.cast(label, tf.int32),depth=label_shape)

        # Reshape image data into the original shape
        image = tf.reshape(image, np.asarray(image_target_size))/255.
        # for the training images, do some data augmentation
        if augmentation:
            image = augment_patch(image)
            image = tf.reshape(image, np.asarray(image_target_size))

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.batch([image, label], batch_size=batch_size, capacity=30,
                                                num_threads=2)

        return images, labels
