# -*- coding:utf-8 -*-
from Seg_model import *
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import easydict
import os
import cv2

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "img_ch": 3,
                           
                           "batch_size": 2,
                           
                           "n_classes": 32,
                           
                           "epochs": 400,
                           
                           "lr": 0.0001,
                           
                           "train": True,
                           
                           "tr_img_path": "D:/[1]DB/[3]detection_DB/CamSeq01",
                           
                           "tr_lab_path": "D:/[1]DB/[3]detection_DB/CamSeq01",

                           "txt_label": "D:/[1]DB/[3]detection_DB/comvid_txt/label_colors.txt",
                           
                           "save_samples": "",
                           
                           "save_checkpoint": "",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": ""})

optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.9, beta_2=0.999)

def func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = tf.io.read_file(lab)
    lab = tf.image.decode_png(lab, 3)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size])
    lab = tf.cast(lab, tf.int32)

    return img, lab

@tf.function
def cal_loss(deeplab_model, images, labels):

    with tf.GradientTape() as tape:

        output = deeplab_model(images, True)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, output)
    grads = tape.gradient(loss, deeplab_model.trainable_variables)
    optim.apply_gradients(zip(grads, deeplab_model.trainable_variables))

    return loss

def main():
    
    deeplab_model = Deeplabv3(input_shape=(256, 256, 3), classes=FLAGS.n_classes-1)  # Deeplab V3 plus model
    deeplab_model.summary()

    tr_images = os.listdir(FLAGS.tr_img_path)
    tr_labels = os.listdir(FLAGS.tr_lab_path)
    tr_img_buf = []
    tr_lab_buf = []
    for i in range(len(tr_images)):
        if len(tr_images[i].split('_')) == 3:
            tr_lab_buf.append(tr_images[i])
        else:
            tr_img_buf.append(tr_images[i])
    tr_lab_buf = [FLAGS.tr_img_path + "/" + data for data in tr_lab_buf]
    tr_img_buf = [FLAGS.tr_img_path + "/" + data for data in tr_img_buf]
    tr_lab_buf = np.array(tr_lab_buf)
    tr_img_buf = np.array(tr_img_buf)

    txt_labels = np.loadtxt(FLAGS.txt_label, dtype=np.int32, skiprows=0, usecols=[0, 1, 2])
    txt_names = np.loadtxt(FLAGS.txt_label, dtype=np.str, skiprows=0, usecols=3)
    
    #################################################

    count = 0
    color_to_label = np.zeros(256**3, dtype=np.int32)
    for i, color in enumerate(txt_labels):
        color_to_label[(color[0] * 256 + color[1]) * 256 + color[2]] = i

    for epoch in range(FLAGS.epochs):
        gener = tf.data.Dataset.from_tensor_slices((tr_img_buf, tr_lab_buf))
        gener = gener.map(func)
        gener = gener.batch(FLAGS.batch_size)
        gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

        it = iter(gener)
        idx = len(tr_img_buf) // FLAGS.batch_size
        for i in range(idx):
            images, labels = next(it)

            labels_np = labels.numpy()
            index = (labels_np[:, :, :, 0] * 256 + labels_np[:, :, :, 1]) * 256 + labels_np[:, :, :, 2]
            labels = color_to_label[index]
            not_one_hot_labels = labels
            labels = tf.one_hot(labels, FLAGS.n_classes - 1)

            loss = cal_loss(deeplab_model, images, labels)

            if count % 10 == 0:
                print(loss)

            if count % 100 == 0:
                output = deeplab_model.predict(tf.expand_dims(images[0], 0))
                pred_mask = tf.argmax(tf.nn.softmax(output, -1), -1)
                pred_mask = tf.squeeze(pred_mask, 0)
                pred_mask = np.array(pred_mask.numpy())

                grounnd_mask = np.array(not_one_hot_labels[0]).astype('uint8')
                grounnd_mask = txt_labels[grounnd_mask]

                pred_mask_color = txt_labels[pred_mask]

                plt.imsave("C:/Users/Yuhwan/Pictures/img/predict_{}.png".format(count),pred_mask_color)
                plt.imsave("C:/Users/Yuhwan/Pictures/img/grounnd_mask_{}.png".format(count),grounnd_mask)

                
                

            count += 1


if __name__ == "__main__":
    main()
