import cv2
import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
from tf_data_handler import write_tfrecord
import config

"""
Extract local regions around a landmark and give the appropriate label
Sample some from the background as well in the process
"""


def make_image_patches(
        rgb_frame,
        frameid,
        locations,
        objects_to_include=[],
        joints=[0],
        patch_size=[28, 28]
):
    patches = []
    labels = []

    for o in range(len(objects_to_include)):
        obj = objects_to_include[o]
        for j in range(len(joints)):
            jnt = joints[j]
            y, x = int(locations[obj][jnt][frameid, 0]), int(locations[obj][jnt][frameid, 1])

            # get the positive example
            patch = rgb_frame[(x - patch_size[0] / 2 - 1):(x + patch_size[0] / 2 - 1),
                    (y - patch_size[1] / 2 - 1):(y + patch_size[1] / 2 - 1), :]
            patches.append(patch)
            labels.append(jnt + 1)

            # also get a corrsponding negative example
            random_noise = np.random.randint(20, size=3)
            bg_patch = rgb_frame[
                       (x + random_noise[0] - patch_size[0] / 2 - 1):(x + random_noise[0] + patch_size[0] / 2 - 1),
                       (y + random_noise[1] - patch_size[1] / 2 - 1):(y + random_noise[1] + patch_size[1] / 2 - 1), :]
            # plt.imshow(bg_patch); plt.show();
            patches.append(bg_patch)
            labels.append(0)
    return patches, labels


"""
load the labels obtained from running Yuliang's tacking system
return parameter: 
   [object][joint][frame,x/y]
"""


def load_joints(video_name,
                label_folder,
                objects_to_include=[],
                joints=[0, 1]):
    assert len(objects_to_include) > 0, 'No objects specified'
    jnts = []

    for object in list(objects_to_include):
        lab = scio.loadmat(os.path.join(label_folder, '%s_obj_%d_seq.mat' % (video_name.split('.')[0], object + 1)))

        jnt_list = []
        for joint in joints:
            xy_locs = np.transpose(np.asarray([lab['x_seq'][joint, :], lab['y_seq'][joint, :]]))
            jnt_list.append(xy_locs)
        jnts.append(jnt_list)
    return jnts


def main(
        video_folder='',
        video_name='',
        label_folder='',
        objects_to_include=None,
        tfrecord_dir='',
        config=None
):
    data_prop = config.data_prop
    jnts = load_joints(video_name=video_name,
                       label_folder=label_folder,
                       objects_to_include=objects_to_include,
                       joints=config.joints_to_extract)

    # data structure to hold patches in memory
    all_patches, all_labels = [], []

    # Load in the video to read
    video_stream = cv2.VideoCapture(os.path.join(video_folder, video_name))
    frameid = 0

    print 'Reading from video...'
    while (video_stream.isOpened()):
        flag, frame = video_stream.read()
        assert flag, 'Reading from video has failed!'

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        patches, labels = make_image_patches(rgb_frame=rgb_frame,
                                             frameid=frameid,
                                             locations=jnts,
                                             objects_to_include=objects_to_include,
                                             joints=config.joints_to_extract,
                                             patch_size=config.image_target_size[:2])
        all_patches.extend(patches)
        all_labels.extend(labels)
        frameid = frameid + 1
        if frameid == 100:
            break

    print 'Finished reading from video. Making train vs test split'

    # import ipdb; ipdb.set_trace();
    total_items = len(all_patches)
    arr = np.arange(total_items)
    np.random.shuffle(arr)

    all_patches = np.asarray(all_patches)[arr]
    all_labels = np.asarray(all_labels)[arr]
    #assert (np.sum(data_prop.values()) != 1.), 'Train vs Test split specified incorrectly'

    train_idx_lim = int(data_prop['train'] * total_items)
    val_idx_lim = int((data_prop['train'] + data_prop['val']) * total_items)

    # write the tf records as specified
    if config.train_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.train_tfrecords,
            all_patches[:train_idx_lim],
            all_labels[:train_idx_lim])

    if config.val_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.val_tfrecords,
            all_patches[train_idx_lim:val_idx_lim],
            all_labels[train_idx_lim:val_idx_lim])

    if config.test_tfrecords != None:
        write_tfrecord(
            tfrecord_dir,
            config.test_tfrecords,
            all_patches[val_idx_lim:],
            all_labels[val_idx_lim:])


if __name__ == '__main__':
    config = config.Config()
    main(
        video_folder=config.base_dir,
        video_name=config.video_name,
        label_folder=config.label_dir,
        objects_to_include=np.asarray(config.objects_to_include),
        tfrecord_dir=config.tfrecord_dir,
        config=config
    )
