import sys
sys.path.append('/home/yiizy/neuron_inject/src/networks/models/research/slim')
import tensorflow as tf
from nets import mobilenet_v1
from datasets import imagenet
import math
import random
import struct
import numpy as np
from preprocessing.preprocessing_factory import get_preprocessing
from tensorflow.python.platform import gfile
import psycopg2

image_set = [
23586, 23204, 28772, 4371, 16582, 4611, 43547, 29932, 8769, 13231, 35387, 25205, 16362, 37600, 2477, 699, 10564, 12496, 7109, 4416, 27509, 17993, 11135, 26298, 40021, 29491, 32590, 4868, 10482, 45033, 35339, 44070, 42836, 35657, 49053, 46130, 20563, 40691, 21284, 15361, 22582, 48499, 40587, 16959, 27034, 16465, 18979, 26866, 39428, 15888, 38959, 36925, 5261, 43962, 32650, 36331, 18842, 5245, 43778, 26669, 893, 28860, 26549, 5488, 14731, 34760, 48221, 18399, 29259, 40324, 44759, 39556, 8518, 30085, 45880, 959, 7190, 5098, 42814, 29167, 5010, 47294, 46591, 33001, 48294, 36806, 14626, 16852, 28171, 4707, 35075, 22628, 4571, 35794, 37751, 44946, 15166, 16110, 32442, 11022, 7221, 8369, 14936, 7800, 17956, 23323, 23679, 32389, 21238, 5066, 49776, 32603, 34761, 23802, 33953, 22893, 3201, 29444, 16059, 18996, 8338, 1133, 31272, 36454, 30982, 29682, 10747, 12576, 15214, 8431, 14466, 5571, 45837, 40283, 45601, 17251, 45622, 21203, 39302, 12731, 46103, 10360, 38136, 4399, 47221, 13788, 34705, 8128, 10058, 44834, 2569, 42609, 12612, 43583, 32788, 21232, 27917, 13843, 49935, 11644, 46292, 46878, 37740, 25165, 15301, 16507, 8412, 20344, 49627, 45970, 7981, 45942, 11581, 24539, 19010, 4900, 22507, 45163, 42494, 22947, 28168, 19110, 41634, 10992, 15268, 37604, 32208, 45599, 37444, 16525, 49169, 38151, 12958, 8237, 8259, 45467, 8480, 17505, 22182, 16636, 3359, 40775, 16028, 43859, 20111, 8095, 48413, 20506, 35654, 11182, 9854, 17375, 10078, 2, 47460, 8211, 34950, 39726, 20545, 35393, 232, 28753, 6609, 10608, 4700, 3952, 30228, 7182, 33644, 4078, 31907, 23788, 34057, 34209, 44987, 630, 39070, 28210, 3960, 38708, 20419, 49332, 28430, 44057, 1962, 46306, 22960, 32518, 27208, 38283, 21366, 30285, 982, 7749, 45710, 42019, 45141, 49307, 24443, 26494, 16820, 46688, 27680, 44948, 29013, 10671, 44623, 16963, 2886, 47515, 20244, 32925, 31290, 32699, 33078, 23588, 10622, 27792, 28307, 12791, 23694, 31374, 38858, 32653, 47043, 20226, 24491, 3937, 37808, 29962, 3748, 42725, 3448, 8012, 39183, 47679, 29065, 42078, 39719, 19908, 24775, 26363, 12797, 13569, 44748, 11100, 43857, 20415, 16162, 18514, 45827, 20915, 43245, 11612, 15273, 48323, 39129, 32963, 25541, 34709, 4990, 577, 26, 43989, 25103, 5971, 49707, 29774, 5491, 4756, 35781, 13440, 42907, 29388, 39426, 10246, 28217, 21601, 27113, 1451, 18553, 27600, 48948, 12764, 5988, 15752, 35086, 28936, 26461, 7136, 47626, 42457, 44439, 1765, 23112, 24698, 35184, 1442, 46873, 21800, 18116, 19731, 45143, 4184, 1122, 18878, 17247, 19259, 39532, 13907, 25765, 37867, 34789, 21084, 4468, 25704, 15274, 29380, 25558, 24968, 42080, 24880, 47071, 32947, 35362, 1381, 38252, 49876, 39210, 44510, 24398, 32845, 34076, 29352, 34980, 33368, 46789, 31899, 40044, 37309
]

target_layer_list = [
'MobilenetV1/MobilenetV1/Conv2d_0',
'MobilenetV1/MobilenetV1/Conv2d_1_depthwise',
'MobilenetV1/MobilenetV1/Conv2d_1_pointwise',
'MobilenetV1/MobilenetV1/Conv2d_4_depthwise',
'MobilenetV1/MobilenetV1/Conv2d_4_pointwise',
'MobilenetV1/MobilenetV1/Conv2d_7_depthwise',
'MobilenetV1/MobilenetV1/Conv2d_7_pointwise',
'MobilenetV1/MobilenetV1/Conv2d_11_depthwise',
'MobilenetV1/MobilenetV1/Conv2d_11_pointwise',
'MobilenetV1/MobilenetV1/Conv2d_13_depthwise',
'MobilenetV1/MobilenetV1/Conv2d_13_pointwise'
]

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

# Get layer's 1% of neuron number
def get_one_percent(layer):
    _, l_h, l_w, l_c = layer_size_dict[layer]
    return round(l_h * l_w * l_c * 0.01)

def int_set_to_txt(input_list):
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

def bin2fp32(total_bin):
    assert len(total_bin) == 32
    return struct.unpack('!f',struct.pack('!I', int(total_bin, 2)))[0]

def bin2fp16(total_bin):
    assert len(total_bin) == 16
    sign_bin = total_bin[0]
    if sign_bin == '0':
        sign_val = 1.0
    else:
        sign_val = -1.0
    exponent_bin = total_bin[1:6]
    mantissa_bin = total_bin[6:]
    assert len(mantissa_bin) == 10
    exponent_val = int(exponent_bin,2)
    mantissa_val = 0.0
    for i in range(10):
        if mantissa_bin[i] == '1':
            mantissa_val += pow(2,-i-1)
    # Handling subnormal numbers
    if exponent_val == 0:
        return sign_val * pow(2,-14) * mantissa_val
    # Handling normal numbers
    else:
        return sign_val * pow(2,exponent_val-15) * (1 + mantissa_val)

def dispatch_single_layer():
    try:
        conn = psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')
    for image in image_set:
        for layer in target_layer_list:
            #for prec in ['int8', 'fp16', 'fp32']:
            for prec in ['int8']:
                for num_inj in [1, 10, 100, get_one_percent(layer)]:
                    _, l_h, l_w, l_c = layer_size_dict[layer]

                    inj_h_set = []
                    inj_w_set = []
                    inj_c_set = []
                    inj_tup_set = []
                    while len(inj_tup_set) != num_inj:
                        tup = (random.randint(0,l_h-1), random.randint(0,l_w-1), random.randint(0,l_c-1))
                        if tup not in inj_tup_set:
                            inj_tup_set.append(tup)
                            inj_h_set.append(tup[0])
                            inj_w_set.append(tup[1])
                            inj_c_set.append(tup[2])

                    inj_h_set_txt = int_set_to_txt(inj_h_set)
                    inj_w_set_txt = int_set_to_txt(inj_w_set)
                    inj_c_set_txt = int_set_to_txt(inj_c_set)

                    if prec == 'fp32':
                        delta_fp_set = []
                        for _ in range(num_inj):
                            one_bin = ''
                            for _ in range(32):
                                one_bin += str(np.random.randint(0,2))
                            delta_fp_set.append(bin2fp32(one_bin))
                        delta_set_txt = int_set_to_txt(delta_fp_set)

                    elif prec == 'fp16':
                        delta_fp_set = []
                        for _ in range(num_inj):
                            one_bin = ''
                            for _ in range(16):
                                one_bin += str(np.random.randint(0,2))
                            delta_fp_set.append(bin2fp16(one_bin))
                        delta_set_txt = int_set_to_txt(delta_fp_set)

                    elif prec == 'int8':
                        delta_int8_set = np.random.randint(-pow(2,21),pow(2,21), size=num_inj)
                        delta_set_txt = int_set_to_txt(delta_int8_set)

                    else:
                        print('Error!')
                        exit(-1)

                    exe_str = """INSERT INTO mb_{}_random_inject (job_status, image_id, layer, num_inj, inj_h_set, inj_w_set, inj_c_set, delta_set) VALUES (\'{}\',{},\'{}\',{},{},{},{},{});""".format(prec, 'READY', image, layer, num_inj, inj_h_set_txt, inj_w_set_txt, inj_c_set_txt, delta_set_txt) 
                    cur = conn.cursor()
                    cur.execute(exe_str)
                    conn.commit()


def dispatch_all_layers():
    try:
        conn = psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')
    for image in image_set:
        for prec in ['int8', 'fp16', 'fp32']:
            for num_inj in ['1perc']:
                for _ in range(100):
                    inj_h_set = []
                    inj_w_set = []
                    inj_c_set = []
                    delta_val_set = []
                    inj_count = 0

                    for layer in layer_size_dict:
                        # First decide the injection positions
                        _, l_h, l_w, l_c = layer_size_dict[layer]
                        layer_num_inj = get_one_percent(layer)
                        inj_count += layer_num_inj
                        layer_inj_tup_set = []
                        while len(layer_inj_tup_set) != layer_num_inj:
                            tup = (random.randint(0,l_h-1), random.randint(0,l_w-1), random.randint(0,l_c-1))
                            if tup not in layer_inj_tup_set:
                                layer_inj_tup_set.append(tup)
                                inj_h_set.append(tup[0])
                                inj_w_set.append(tup[1])
                                inj_c_set.append(tup[2])

                        if prec == 'fp32':
                            for _ in range(layer_num_inj):
                                one_bin = ''
                                for _ in range(32):
                                    one_bin += str(np.random.randint(0,2))
                                delta_val_set.append(bin2fp32(one_bin))

                        elif prec == 'fp16':
                            for _ in range(layer_num_inj):
                                one_bin = ''
                                for _ in range(16):
                                    one_bin += str(np.random.randint(0,2))
                                delta_val_set.append(bin2fp16(one_bin))

                        elif prec == 'int8':
                            layer_delta_int8_set = np.random.randint(-pow(2,21),pow(2,21), size=layer_num_inj)
                            for elem in layer_delta_int8_set:
                                delta_val_set.append(elem)

                        else:
                            print('Error!')
                            exit(-1)

                    assert inj_count == len(inj_h_set) == len(inj_w_set) == len(inj_c_set) == len(delta_val_set)

                    inj_h_set_txt = int_set_to_txt(inj_h_set)
                    inj_w_set_txt = int_set_to_txt(inj_w_set)
                    inj_c_set_txt = int_set_to_txt(inj_c_set)
                    delta_set_txt = int_set_to_txt(delta_val_set)

                    exe_str = """INSERT INTO mb_{}_random_inject (job_status, image_id, layer, num_inj, inj_h_set, inj_w_set, inj_c_set, delta_set) VALUES (\'{}\',{},\'{}\',{},{},{},{},{});""".format(prec, 'READY', image, '1perc', -1, inj_h_set_txt, inj_w_set_txt, inj_c_set_txt, delta_set_txt)
                    cur = conn.cursor()
                    cur.execute(exe_str)
                    conn.commit()



#dispatch_single_layer()
dispatch_all_layers()
