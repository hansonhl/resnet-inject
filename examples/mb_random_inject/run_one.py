#!/bin/python3
import sys
sys.path.append('/home/yiizy/neuron_inject/src/networks/models/research/slim')
import tensorflow as tf
import math
import psutil
import os
from datasets import imagenet
import random
import numpy as np
from preprocessing.preprocessing_factory import get_preprocessing
import psycopg2

layer_size_dict = {
'MobilenetV1/MobilenetV1/Conv2d_0' : (1, 112, 112, 32),
'MobilenetV1/MobilenetV1/Conv2d_1_depthwise' : (1, 112, 112, 32),
'MobilenetV1/MobilenetV1/Conv2d_1_pointwise' : (1, 112, 112, 64),
'MobilenetV1/MobilenetV1/Conv2d_2_depthwise' : (1, 56, 56, 64),
'MobilenetV1/MobilenetV1/Conv2d_2_pointwise' : (1, 56, 56, 128),
'MobilenetV1/MobilenetV1/Conv2d_3_depthwise' : (1, 56, 56, 128),
'MobilenetV1/MobilenetV1/Conv2d_3_pointwise' : (1, 56, 56, 128),
'MobilenetV1/MobilenetV1/Conv2d_4_depthwise' : (1, 28, 28, 128),
'MobilenetV1/MobilenetV1/Conv2d_4_pointwise' : (1, 28, 28, 256),
'MobilenetV1/MobilenetV1/Conv2d_5_depthwise' : (1, 28, 28, 256),
'MobilenetV1/MobilenetV1/Conv2d_5_pointwise' : (1, 28, 28, 256),
'MobilenetV1/MobilenetV1/Conv2d_6_depthwise' : (1, 14, 14, 256),
'MobilenetV1/MobilenetV1/Conv2d_6_pointwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_7_depthwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_7_pointwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_8_depthwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_8_pointwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_9_depthwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_9_pointwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_10_depthwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_10_pointwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_11_depthwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_11_pointwise' : (1, 14, 14, 512),
'MobilenetV1/MobilenetV1/Conv2d_12_depthwise' : (1, 7, 7, 512),
'MobilenetV1/MobilenetV1/Conv2d_12_pointwise' : (1, 7, 7, 1024),
'MobilenetV1/MobilenetV1/Conv2d_13_depthwise' : (1, 7, 7, 1024),
'MobilenetV1/MobilenetV1/Conv2d_13_pointwise' : (1, 7, 7, 1024)
}

def list_to_db_array(input_list):
    out_txt = 'ARRAY['
    for n in range(len(input_list) - 1):
        if math.isnan(input_list[n]):
            out_txt += '\'nan\'::double precision,'
        elif math.isinf(input_list[n]):
            if input_list[n] > 0:
                out_txt += '\'infinity\'::double precision,'
            else:
                out_txt += '\'-infinity\'::double precision,'
        else:
            out_txt += str(input_list[n]) + ','

    if math.isnan(input_list[-1]):
        out_txt += '\'nan\'::double precision]'
    elif math.isinf(input_list[-1]):
        if input_list[-1] > 0:
            out_txt += '\'infinity\'::double precision]'
        else:
            out_txt += '\'-infinity\'::double precision]'
    else:
        out_txt += str(input_list[-1]) + ']'
    return out_txt

# De-quantizing a set of int8 values to float values
def int8_set_to_fp_set(int8_set):
    fp_set = []
    for num in int8_set:
        fp_set.append(num * ((5.9997616 - 0) / 255))
    return fp_set

def hwc_set_to_pos_set(h_set, w_set, c_set):
    pos_set = []
    assert len(h_set) == len(w_set) == len(c_set)

    for n in range(len(h_set)):
        pos_set.append((h_set[n], w_set[n], c_set[n]))
    return pos_set

def run_one(prec):
    assert prec in ['int8','fp16','fp32']

    if prec  == 'fp16':
        from nets import mobilenet_v1_fp16
    else:
        from nets import mobilenet_v1

    try:
        conn = psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')

    # Get one job
    cur = conn.cursor()
    cur.execute("""UPDATE mb_{}_random_inject SET job_status = 'RUNNING' WHERE inj_id = (SELECT inj_id FROM mb_{}_random_inject WHERE job_status = 'READY' LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING inj_id, image_id, layer, num_inj, inj_h_set, inj_w_set, inj_c_set, delta_set;""".format(prec, prec))

    job = cur.fetchall()[0]
    conn.commit()

    inj_id = job[0]
    image_id = job[1]
    layer = job[2]
    num_inj = job[3]
    inj_h_set = job[4]
    inj_w_set = job[5]
    inj_c_set = job[6]
    delta_set = job[7]

    if prec == 'int8':
        delta_fp_set = int8_set_to_fp_set(delta_set)
    else:
        delta_fp_set = delta_set

    inj_pos_set = hwc_set_to_pos_set(inj_h_set, inj_w_set, inj_c_set)
    
    if prec in ['fp32','fp16']:
        model_dir = '/home/yiizy/neuron_inject/src/networks/mobilenet_v1_normal/'
        checkpoint = model_dir + 'mobilenet_v1_1.0_224.ckpt'
    elif prec == 'int8':
        model_dir = '/home/yiizy/neuron_inject/src/networks/mobilenet_v1_quant/'
        checkpoint = model_dir + 'mobilenet_v1_1.0_224_quant.ckpt'
    else:
        print('Error!')
        exit(12)

    tf.reset_default_graph()

    file_input = tf.placeholder(tf.string, ())
    image = tf.image.decode_jpeg(tf.read_file(file_input), channels=3)
    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128.  - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (224, 224))

    if prec == 'fp16':
        images = tf.cast(images, tf.float16)
        with tf.contrib.slim.arg_scope(mobilenet_v1_fp16.mobilenet_v1_arg_scope(is_training=False)):
            logits, endpoints = mobilenet_v1_fp16.mobilenet_v1(images, num_classes=1001, is_training=False, depth_multiplier=1.0, bound_type = 'RD', inj_layer=layer, inj_pos=inj_pos_set)

    else:
        with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
            logits, endpoints = mobilenet_v1.mobilenet_v1(images, num_classes=1001, is_training=False, depth_multiplier=1.0, bound_type = 'RD', inj_layer=layer, inj_pos=inj_pos_set) 

    if prec == 'int8':
        tf.contrib.quantize.create_eval_graph()

    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    if prec == 'fp16':
        saver = tf.train.Saver(var_list = [v for v in all_variables if 'Logits' not in v.name and 'delta' not in v.name])
    else:
        saver = tf.train.Saver(var_list = [v for v in all_variables if 'delta' not in v.name])

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        if prec == 'fp16':
            for variable in all_variables:
                if 'Logits/dense/kernel' in variable.name:
                    var = tf.contrib.framework.load_variable(checkpoint, 'MobilenetV1/Logits/Conv2d_1c_1x1/weights')
                    sess.run(variable.assign(var[0,0,:,:]))
                if 'Logits/dense/bias' in variable.name:
                    var = tf.contrib.framework.load_variable(checkpoint, 'MobilenetV1/Logits/Conv2d_1c_1x1/biases')
                    sess.run(variable.assign(var))

        # Give delta value
        if num_inj > 0:
            delta_np = np.zeros(shape = layer_size_dict[layer], dtype=np.float32)
            for n_j in range(num_inj):
                delta_np[0][inj_h_set[n_j]][inj_w_set[n_j]][inj_c_set[n_j]] = delta_fp_set[n_j]
            with tf.variable_scope('MobilenetV1',reuse = True):
                sess.run(tf.get_variable('delta',trainable = False).assign(delta_np))
        elif num_inj == -1:
            start_seq = 0
            num_delta = 0
            for layer in layer_size_dict:
                layer_num_inj = get_one_percent(layer)
                delta_np = np.zeros(shape = layer_size_dict[layer], dtype=np.float32)
                for n_j in range(layer_num_inj):
                    inj_seq = start_seq + n_j
                    delta_np[0][inj_h_set[inj_seq]][inj_w_set[inj_seq]][inj_c_set[inj_seq]] = delta_fp_set[inj_seq]
                with tf.variable_scope('MobilenetV1',reuse = True):
                    if num_delta == 0:
                        sess.run(tf.get_variable('delta'.format(),trainable = False).assign(delta_np))
                    else:
                        sess.run(tf.get_variable('delta_{}'.format(num_delta).format(),trainable = False).assign(delta_np))
                num_delta += 1
                start_seq += layer_num_inj
        else:
            print('Can not handle this number of errors!')
            exit(21)

        probs, bf_sftmaxes = sess.run([endpoints['Predictions'], endpoints['Logits']], feed_dict={file_input: '/home/yiizy/val_images/ILSVRC2012_val_' + str(image_id).zfill(8) + '.JPEG'})
        top5_labels = probs[0].argsort()[::-1][:5]
        top5_probs = probs[0][top5_labels]
        top5_bf_sftmaxes = bf_sftmaxes[0][top5_labels]

        top5_labels_txt = list_to_db_array(top5_labels)
        top5_probs_txt = list_to_db_array(top5_probs)
        top5_bf_sftmaxes_txt = list_to_db_array(top5_bf_sftmaxes)

        delta_txt = list_to_db_array(delta_fp_set)

        if prec == 'int8':
            exe_text = """UPDATE mb_{}_random_inject SET job_status = 'DONE', inj_prob = {}, inj_bf_sftmax = {}, inj_label = {}, delta_fp_set = {} WHERE inj_id = {};""".format(prec, top5_probs_txt, top5_bf_sftmaxes_txt, top5_labels_txt, delta_txt, inj_id)
        else:
            exe_text = """UPDATE mb_{}_random_inject SET job_status = 'DONE', inj_prob = {}, inj_bf_sftmax = {}, inj_label = {} WHERE inj_id = {};""".format(prec, top5_probs_txt, top5_bf_sftmaxes_txt, top5_labels_txt, inj_id)


        cur.execute(exe_text)
        conn.commit()
        conn.close()
    sess.close()
    tf.reset_default_graph()

#run_one('int8')
