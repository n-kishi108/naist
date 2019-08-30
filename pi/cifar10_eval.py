# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
 
"""Evaluation for CIFAR-10.
 
Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.
 
Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.
 
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
 
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
from datetime import datetime
import math
import time
 
import numpy as np
import tensorflow as tf
 
from tensorflow.models.image.cifar10 import cifar10
 
FLAGS = tf.app.flags.FLAGS
 
# 評価結果の格納先ディレクトリ
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
 
# 評価用データ
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
 
# チェックポイント格納先ディレクトリ（訓練済データの格納先を指定）
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
 
# 繰り返し実行する場合の実行間隔 (秒)
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
 
# サンプリング数
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
 
# 一度だけ実行 (True) または、繰り返し実行 (False) を指定
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")
 
 
#
# 評価を 1 回分実行
#
def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # チェックポイントを復元
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
 
        # Queue Runner (キューによる実行) を開始
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
 
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # 正しく識別できた数を記録
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
 
            # 識別率 (精度) を計算
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
 
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
 
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
 
#
# 評価を実行
#
def evaluate():
    with tf.Graph().as_default() as g:
        # 検証用データとラベルを取得
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)
 
        # 予測モデルをグラフとして構築
        logits = cifar10.inference(images)
 
        # 予測結果から精度 (一致または不一致の割合) を計算
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
 
        # 学習した変数の移動平均を復元
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
 
        # Tensorflow のグラフのサマリを行う処理
        summary_op = tf.merge_all_summaries()
 
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)
 
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
 
 
def main(argv=None):  # pylint: disable=unused-argument
    # データセットをダウンロードし、解凍
    cifar10.maybe_download_and_extract()
 
    # 訓練済データが存在する場合、削除
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
 
    # 訓練済データ格納先フォルダを作成
    tf.gfile.MakeDirs(FLAGS.eval_dir)
 
    # 評価を実行
    evaluate()
 
 
if __name__ == '__main__':
    tf.app.run()
