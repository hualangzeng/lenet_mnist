import tensorflow as tf
#from lenet5.lenet5_lib import *
from lenet import lenet5_parameter
import numpy as np
#import lenet5_parameter

def create_weight(shape):
    weight = tf.Variable(tf.truncated_normal(shape,  stddev=0.1), name="weight")
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(lenet5_parameter.REGULARIZE_RATE)(weight))
    return weight

def create_bias(shape):
    bias = tf.Variable(tf.constant(0.1, shape = shape))
    return bias

def conv(input, weight, bias, stride = [1,1,1,1], padding_value = "VALID", activation_function = None):
    conv_result = tf.nn.conv2d(input, weight, strides=stride, padding=padding_value)
    result = tf.nn.bias_add(conv_result, bias)
    if activation_function is not None:
        result = activation_function(result)

    return result
def pooling(input, ksize = [1,2,2,1], stride = [1,2,2,1], padding_value = "VALID"):
    result = tf.nn.max_pool(input, ksize=ksize, strides=stride, padding=padding_value)
    return result


def fc_layer(input, weight, bias, if_drop_out = None, activation_function = None):
    result = tf.nn.bias_add(tf.matmul(input, weight),  bias)
    if activation_function is not None:
        result = activation_function(result)
    #if if_drop_out is not None:
     #   return tf.nn.dropout(result, 0.8)

    return result

def lenet5_model(input):
    #conv_1
    conv1_weight = create_weight(lenet5_parameter.conv1_weight_size)
    conv1_bias = create_bias(lenet5_parameter.conv1_bias_size)
    conv1_output = conv(input, conv1_weight, conv1_bias, lenet5_parameter.conv1_stride_size,lenet5_parameter.conv1_padding_value,activation_function=tf.nn.relu)

    #pooling_2
    pooling2_output = pooling(conv1_output, ksize=lenet5_parameter.pooling2_k_size, stride=lenet5_parameter.pooling2_stride_size,padding_value=lenet5_parameter.pooling2_padding_value)

    #conv_3
    conv3_weight = create_weight(lenet5_parameter.conv3_weight_size)
    conv3_bias = create_bias(lenet5_parameter.conv3_bias_size)
    conv3_output = conv(pooling2_output, conv3_weight, conv3_bias,
                        lenet5_parameter.conv3_stride_size,lenet5_parameter.conv3_padding_value,tf.nn.relu)

    # pooling_4
    pooling4_output = pooling(conv3_output, ksize=lenet5_parameter.pooling4_k_size,
                              stride=lenet5_parameter.pooling4_stride_size,
                              padding_value=lenet5_parameter.pooling4_padding_value)

    # conv_5
    conv5_weight = create_weight(lenet5_parameter.conv5_weight_size)
    conv5_bias = create_bias(lenet5_parameter.conv5_bias_size)
    conv5_output = conv(pooling4_output, conv5_weight, conv5_bias, lenet5_parameter.conv5_stride_size,
                        lenet5_parameter.conv5_padding_value, tf.nn.relu)

    #conv_5 reshape
    conv5_output_reshape = tf.reshape(conv5_output, lenet5_parameter.conv5_reshape_size)

    # fc_6
    fc_6_weight = create_weight(lenet5_parameter.fc_6_weight_size)
    fc_6_bias = create_bias(lenet5_parameter.fc_6_bias_size)
    fc_6_output = fc_layer(conv5_output_reshape, fc_6_weight, fc_6_bias, activation_function=tf.nn.relu)


    # fc_18
    fc_7_weight = create_weight(lenet5_parameter.fc_7_weight_size)
    fc_7_bias = create_bias(lenet5_parameter.fc_7_bias_size)
    fc_7_output = fc_layer(fc_6_output, fc_7_weight, fc_7_bias, activation_function=tf.nn.relu)

    return fc_7_output









