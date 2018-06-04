
BATCH_SIZE = 100
IMAGE_SIZE = [28,28,1]
TRAIN_SIZE= [BATCH_SIZE, 32,32,1]
OUTPUT_SIZE = [BATCH_SIZE, 10]
IMAGE_CHANNEL_NUM = 1
REGULARIZE_RATE = 1e-5
INITIAL_LEARNING_RATE = 0.1

conv1_weight_size = [5,5,1,6]
conv1_bias_size = [6]
conv1_stride_size = [1,1,1,1]
conv1_padding_value = "VALID"


pooling2_k_size = [1, 2, 2, 1]
pooling2_stride_size = [1, 2,2,1]
pooling2_padding_value = "VALID"

conv3_weight_size = [5,5,6,16]
conv3_bias_size = [16]
conv3_stride_size = [1,1,1,1]
conv3_padding_value = "VALID"



pooling4_k_size = [1, 2, 2, 1]
pooling4_stride_size = [1, 2,2,1]
pooling4_padding_value = "VALID"

conv5_weight_size = [5,5,16,120]
conv5_bias_size = [120]
conv5_stride_size = [1,1,1,1]
conv5_padding_value = "VALID"

conv5_reshape_size = [BATCH_SIZE, 120]

fc_6_weight_size = [120, 84]
fc_6_bias_size = [84]



fc_7_weight_size = [84, 10]
fc_7_bias_size = [10]








