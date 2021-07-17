# -*- coding:utf-8 -*-
from Seg_model import *
from SEg_model_2 import *
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
                           
                           "tr_img_path": "D:/[1]DB/[3]detection_DB/camvid/CamVid/train",
                           
                           "tr_lab_path": "D:/[1]DB/[3]detection_DB/camvid/CamVid/train_labels",

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

def compute_miou(actual, pred):
    a = actual
    a = a.reshape((FLAGS.img_size * FLAGS.img_size,))
    a_count = np.bincount(a, weights = None, minlength = FLAGS.n_classes) # A
    
    b = pred
    b = b.reshape((FLAGS.img_size * FLAGS.img_size,))
    b_count = np.bincount(b, weights = None, minlength = FLAGS.n_classes) # B
    
    c = (a * (FLAGS.n_classes)) + b   # category
    cm = np.bincount(c, weights = None, minlength = (FLAGS.n_classes) * (FLAGS.n_classes))
    cm = cm.reshape((FLAGS.n_classes, FLAGS.n_classes))
    #m = tf.metrics.MeanIoU(FLAGS.n_classes) # 내일 다시!
    #m = m.update_state(actual, pred)
    #cm = m.numpy()
    
    Nr = np.diag(cm) # A ⋂ B
    Dr = a_count + b_count - Nr # A ⋃ B
    individual_iou = Nr/Dr
    miou = np.nanmean(individual_iou)
    
    return miou

def main():
    #deeplab_model = unet_model(FLAGS.n_classes)
    deeplab_model = Deeplabv3(input_shape=(256, 256, 3), classes=FLAGS.n_classes)  # Deeplab V3 plus model
    deeplab_model.summary()

    tr_images = os.listdir(FLAGS.tr_img_path)
    tr_images = [FLAGS.tr_img_path + "/" + data for data in tr_images]
    tr_labels = os.listdir(FLAGS.tr_lab_path)
    tr_labels = [FLAGS.tr_lab_path + "/" + data for data in tr_labels]
    tr_lab_buf = np.array(tr_labels)
    tr_img_buf = np.array(tr_images)

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

        total_miou = 0.
        a_count = 0
        b_count = 0
        cm = 0
        for step in range(idx):
            images, labels = next(it)

            labels_np = labels.numpy()
            index = (labels_np[:, :, :, 0] * 256 + labels_np[:, :, :, 1]) * 256 + labels_np[:, :, :, 2]
            labels = color_to_label[index]
            not_one_hot_labels = labels
            labels = tf.one_hot(labels, FLAGS.n_classes)

            lab = labels.numpy()
            lab[:, :, :, -1] = 0.

            loss = cal_loss(deeplab_model, images, lab)

            tmp_miou = 0.
            for i in range(FLAGS.batch_size):
                predict = deeplab_model(tf.expand_dims(images[i], 0), False)
                #predict = deeplab_model.predict(tf.expand_dims(images[0], 0))
                predict = tf.argmax(tf.nn.softmax(predict, -1), -1)
                predict = tf.squeeze(predict, 0)
                #predict  = tf.one_hot(predict, FLAGS.n_classes)
                predict = np.array(predict.numpy())

                grounnd_mask = tf.argmax(labels[i], -1)
                grounnd_mask = np.array(grounnd_mask.numpy())
                
                tmp_miou += compute_miou(grounnd_mask, predict)
   

            total_miou += tmp_miou
            

            if count % 10 == 0:
                print("Epoch = {}[{}/{}] loss = {}".format(epoch, step+1, idx, loss))
                #print("( [{}/{}] images's miou = {} )".format((step + 1) * FLAGS.batch_size, 
                #                                               idx * FLAGS.batch_size, 
                #                                               total_miou / ((step + 1) * FLAGS.batch_size)))

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

        #total_miou /= len(tr_img_buf)
        print("( {} epoch's miou = {} )".format(epoch + 1,
                                                total_miou / len(tr_img_buf)))

if __name__ == "__main__":
    main()
