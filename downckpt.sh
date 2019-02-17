#!/bin/sh
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=./checkpoints

# Where the dataset is saved to.
DATASET_DIR=./flowers

# Checkpoint name
CHECKPOINT_NAME=resnet_v2_50_2017_04_14.tar.gz

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt ]; then
  wget http://download.tensorflow.org/models/${CHECKPOINT_NAME}
  tar -xvf ${CHECKPOINT_NAME}
  mv resnet_v2_50.ckpt ${PRETRAINED_CHECKPOINT_DIR}/resnet_v2_50.ckpt
  rm ${CHECKPOINT_NAME}
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}
