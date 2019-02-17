#!/bin/sh
DATASET_DIR=./flowers
TRAIN_DIR=./flowers-models
CHECKPOINT_PATH=./checkpoints/resnet_v2_50.ckpt

if [ ! -d "$TRAIN_DIR" ]; then
  mkdir ${TRAIN_DIR}
fi

# python train_image_classifier.py \
#     --train_dir=${TRAIN_DIR} \
#     --dataset_dir=${DATASET_DIR} \
#     --dataset_name=flowers \
#     --dataset_split_name=train \
#     --model_name=resnet_v2_50 \
#     --checkpoint_path=${CHECKPOINT_PATH} \
#     --checkpoint_exclude_scopes=resnet_v2_50/logits,resnet_v2_50/spatial_squeeze,resnet_v2_50/predictions \
#     --trainable_scopes=resnet_v2_50/logits,resnet_v2_50/spatial_squeeze,resnet_v2_50/predictions \
#     --max_number_of_steps=2000 \
#     --batch_size=32 \
#     --learning_rate=0.0001 \
#     --learning_rate_decay_type=fixed \
#     --save_interval_secs=60 \
#     --save_summaries_secs=60 \
#     --log_every_n_steps=100 \
#     --optimizer=rmsprop \
#     --weight_decay=0.00004 \
#     --clone_on_cpu

echo "-----------------------------------------------"
echo "-----------------------------------------------"

  # Run evaluation.
python eval_image_classifier.py \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v2_50 \
    --preprocessing_name inception \
    --eval_image_size 299

echo "-----------------------------------------------"
echo "-----------------------------------------------"

# Fine-tune all the new layers for 500 steps.
# python train_image_classifier.py \
#     --train_dir=${TRAIN_DIR}/all \
#     --dataset_name=flowers \
#     --dataset_split_name=train \
#     --dataset_dir=${DATASET_DIR} \
#     --model_name=resnet_v2_50 \
#     --checkpoint_path=${TRAIN_DIR} \
#     --max_number_of_steps=2000 \
#     --batch_size=64 \
#     --learning_rate=0.0001 \
#     --learning_rate_decay_type=fixed \
#     --save_interval_secs=60 \
#     --save_summaries_secs=60 \
#     --log_every_n_steps=10 \
#     --optimizer=rmsprop \
#     --weight_decay=0.00004 \
#     --clone_on_cpu

echo "-----------------------------------------------"
echo "-----------------------------------------------"

# # Run evaluation.
# python eval_image_classifier.py \
#     --checkpoint_path=${TRAIN_DIR}/all \
#     --eval_dir=${TRAIN_DIR}/all \
#     --dataset_name=flowers \
#     --dataset_split_name=validation \
#     --dataset_dir=${DATASET_DIR} \
#     --model_name=resnet_v2_50
# 
# echo "-----------------------------------------------"
# echo "-----------------------------------------------"
