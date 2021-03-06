# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    inputs = slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                      padding='SAME', scope=scope)
    return inputs
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], )
    inputs = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)
    return inputs


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       store_non_strided_activations=False,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    store_non_strided_activations: If True, we compute non-strided (undecimated)
      activations at the last unit of each block and store them in the
      `outputs_collections` before subsampling them. This gives us access to
      higher resolution intermediate activations which are useful in some
      dense prediction problems but increases 4x the computation and memory cost
      at the last unit of each block.
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      block_stride = 1
      for i, unit in enumerate(block.args):
        if store_non_strided_activations and i == len(block.args) - 1:
          # Move stride from the block's last unit to the end of the block.
          block_stride = unit.get('stride', 1)
          unit = dict(unit, stride=1)

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
            rate *= unit.get('stride', 1)

          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('stride', 1)
            if output_stride is not None and current_stride > output_stride:
              raise ValueError('The target output_stride cannot be reached.')

      # Collect activations at the block's end before performing subsampling.
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

      # Subsampling of the block's output activations.
      if output_stride is not None and current_stride == output_stride:
        rate *= block_stride
      else:
        net = subsample(net, block_stride)
        current_stride *= block_stride
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay, #0.997 by default
      'epsilon': batch_norm_epsilon, #1e-5 by default
      'scale': batch_norm_scale, #True by default
      'updates_collections': batch_norm_updates_collections, #tf.GraphKeys.UPDATE_OPS
      'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def dropout_batch_norm(inputs,
                       decay=0.999, #0.997
                       center=True,
                       scale=False, #true
                       epsilon=0.001, #1e-5
                       activation_fn=None, #relu
                       param_initializers=None,
                       param_regularizers=None,
                       updates_collections=tf.GraphKeys.UPDATE_OPS,
                       is_training=True,
                       reuse=None,
                       variables_collections=None,
                       outputs_collections=None,
                       trainable=True,
                       batch_weights=None,
                       fused=None,
                       data_format='NHWC',
                       zero_debias_moving_mean=False,
                       scope=None,
                       renorm=False,
                       renorm_clipping=None,
                       renorm_decay=0.99,
                       adjustment=None):

  my_scope_name = "BatchNorm"

  output = slim.batch_norm(
    inputs=inputs,
    decay=decay,
    center=center,
    scale=scale,
    epsilon=epsilon,
    activation_fn=activation_fn,
    param_initializers=param_initializers,
    param_regularizers=param_regularizers,
    updates_collections=updates_collections,
    is_training=is_training,
    reuse=tf.AUTO_REUSE,
    variables_collections=variables_collections,
    outputs_collections=outputs_collections,
    trainable=trainable,
    batch_weights=batch_weights,
    fused=fused,
    data_format=data_format,
    zero_debias_moving_mean=zero_debias_moving_mean,
    scope=my_scope_name, #changed
    renorm=renorm,
    renorm_clipping=renorm_clipping,
    renorm_decay=renorm_decay,
    adjustment=adjustment)

  return batch_norm_dropout(my_scope_name, output, 2., outputs_collections)


def batch_norm_clipped(batch_norm_scope, output, scale, activation_fn, outputs_collections):
  with tf.variable_scope(batch_norm_scope, reuse=True) as sc:
    mean = tf.get_variable('moving_mean')
    variance = tf.get_variable('moving_variance')
    gamma = tf.get_variable('gamma')
    beta = tf.get_variable('beta')
    stddev = tf.sqrt(variance)
    stddev_scale = tf.constant(scale)
    cutoff = tf.add(mean, tf.multiply(stddev, stddev_scale))


def add_hist_summary(name, val):
    summary_op = tf.summary.histogram(name, val, collections=[])
    tf.add_to_collection(tf.GraphKeys.SUMMARIES, summary_op)

def add_scal_summary(name, val):
    summary_op = tf.summary.scalar(name, val, collections=[])
    tf.add_to_collection(tf.GraphKeys.SUMMARIES, summary_op)

def my_variance(v, axes):
  mean = tf.reduce_mean(v, axes)
  var = tf.reduce_mean(tf.square(tf.subtract(v, mean)), axes)
  return var


def batch_norm_dropout(batch_norm_scope, output, scale, activation_fn, outputs_collections):
  with tf.variable_scope(batch_norm_scope, reuse=True) as sc:
    mean = tf.get_variable('moving_mean')
    variance = tf.get_variable('moving_variance')
    gamma = tf.get_variable('gamma')
    beta = tf.get_variable('beta')

    axes = [0,1,2]

    add_hist_summary('Mean_before_dropout', tf.reduce_mean(output, axes))
    add_hist_summary('Var_before_dropout', my_variance(output, axes))

    stddev = tf.sqrt(variance)

    stddev_scale = tf.constant(scale)

    cutoff = tf.add(mean, tf.multiply(stddev, stddev_scale))
    reduced_y = tf.divide(tf.subtract(output, beta), gamma)
    dropout = tf.maximum(0., tf.sign(tf.subtract(cutoff, reduced_y)))
    dropped_count = tf.subtract(tf.size(dropout), tf.count_nonzero(dropout, dtype=tf.int32))
    add_scal_summary('Dropped_count', dropped_count)
    percentage_dropped = tf.divide(dropped_count, tf.size(dropout), dtype=tf.float32)
    add_scal_summary('percentage_dropped', percentage_dropped)
    output = tf.multiply(output, dropout)

    # summary_op = tf.summary.histogram('After_dropout', output, collections=[])
    # tf.add_to_collection(tf.GraphKeys.SUMMARIES, summary_op)


    add_hist_summary('Mean_after_dropout', tf.reduce_mean(output, axes))
    add_hist_summary('Var_after_dropout', my_variance(output, axes))

    if activation_fn is not None:
        output = activation_fn(output)

    return output



def resnet_batch_dropout_arg_scope(weight_decay=0.0001,
                                   batch_norm_decay=0.997,
                                   batch_norm_epsilon=1e-5,
                                   batch_norm_scale=True,
                                   activation_fn=tf.nn.relu,
                                   use_batch_norm=True,
                                   batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  batch_norm_params = {
      'decay': batch_norm_decay, #0.997 by default
      'epsilon': batch_norm_epsilon, #1e-5 by default
      'scale': batch_norm_scale, #True by default
      'updates_collections': batch_norm_updates_collections, #tf.GraphKeys.UPDATE_OPS
      'fused': None,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=activation_fn,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc


def collect_named_outputs(collections, alias, outputs):
  """Add `Tensor` outputs tagged with alias to collections.
  It is useful to collect end-points or tags for summaries. Example of usage:
  logits = collect_named_outputs('end_points', 'inception_v3/logits', logits)
  assert 'inception_v3/logits' in logits.aliases
  Args:
    collections: A collection or list of collections. If None skip collection.
    alias: String to append to the list of aliases of outputs, for example,
           'inception_v3/conv1'.
    outputs: Tensor, an output tensor to collect
  Returns:
    The outputs Tensor to allow inline call.
  """
  if collections:
    append_tensor_alias(outputs, alias)
    tf.add_to_collections(collections, outputs)
  return outputs

def append_tensor_alias(tensor, alias):
  """Append an alias to the list of aliases of the tensor.
  Args:
    tensor: A `Tensor`.
    alias: String, to add to the list of aliases of the tensor.
  Returns:
    The tensor with a new alias appended to its list of aliases.
  """
  # Remove ending '/' if present.
  if alias[-1] == '/':
    alias = alias[:-1]
  if hasattr(tensor, 'aliases'):
    tensor.aliases.append(alias)
  else:
    tensor.aliases = [alias]
  return tensor
