#
# Script to run one image loaded from the dataset and evaluaet its result
#


import math
import tensorflow as tf

slim = tf.contrib.slim

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory


dataset_dir = '/local/hanson/imagenet'
dataset_name = 'imagenet'
dataset_split_name = 'validation'
model_name = 'resnet_v2_50'
preprocessing_name = 'inception'
eval_image_size = 299
checkpoint_path = 'checkpoints/resnet_v2_50.ckpt'
eval_dir = 'imagenet_eval_results'

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():

        tf_global_step = slim.get_or_create_global_step()

        tf.logging.info("Preparing dataset")

        dataset = dataset_factory.get_dataset(
            dataset_name, dataset_split_name, dataset_dir)

        network_fn = nets_factory.get_network_fn(
            model_name, num_classes=dataset.num_classes, is_training=False)

        tf.logging.info("Initializing dataset provider")

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=32,
            common_queue_min=1)

        tf.logging.info("Initialized provider, now getting image and label")

        [image, label] = provider.get(['image', 'label'])

        tf.logging.info("Got image with label %s" % label)

        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch([image, label], batch_size=1)

        logits, _ = network_fn(images)

        variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels)
            # 'Recall_5': slim.metrics.streaming_recall_at_k(
                #logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        num_batches = 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            sess.run(names_to_updates.values())

            metric_values = sess.run(names_to_values.values())
            for metric, value in zip(names_to_values.keys(), metric_values):
                tf.logging.info('Metric %s has value: %f' % (metric, value))
        #
        # tf.logging.info('Evaluating %s' % checkpoint_path)
        #
        # slim.evaluation.evaluate_once(
        #     master='',
        #     checkpoint_path=checkpoint_path,
        #     logdir=eval_dir,
        #     num_evals=num_batches,
        #     eval_op=list(names_to_updates.values()),
        #     variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
