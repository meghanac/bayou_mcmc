# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np
import os

import tensorflow as tf

import logging
import argparse
import sys
import json
import textwrap

from data_extractor.data_loader import Loader
from data_extractor.data_reader import Reader
from trainer_vae.model import Model
from trainer_vae.utils import read_config, dump_config
from tester_vae.tSNE_visualizor.plot import plot


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

HELP = """\
Config options should be given as a JSON file (see config.json for example)
}
"""


def train(clargs):
    if clargs.continue_from is not None:
        config_file = os.path.join(clargs.continue_from, 'config.json')
    else:
        config_file = clargs.config

    with open(config_file) as f:
        config = read_config(json.load(f))
    reader = Reader(clargs)
    reader.save_data(clargs.data)  # '../data_extractor/data/1k_vocab_constraint_min_3-600000'
    loader = Loader(clargs, config)
    model = Model(config)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(os.path.join(clargs.save, 'loss_values.log'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Process id is {}'.format(os.getpid()))
    logger.info('GPU device is {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info('Amount of data used is {}\n\t'.format(config.num_batches * config.batch_size))

    analysis_file = open(os.path.join(clargs.save, 'analysis.txt'), 'w+')

    with tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(log_device_placement=False,
                                            allow_soft_placement=True)) as sess:

        saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=1)
        tf.global_variables_initializer().run()

        # restore model
        if clargs.continue_from is not None:
            vars_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            old_saver = tf.compat.v1.train.Saver(vars_)
            ckpt = tf.train.get_checkpoint_state(clargs.continue_from)
            old_saver.restore(sess, ckpt.model_checkpoint_path)

        # training
        for i in range(config.num_epochs):
            loader.reset_batches()
            avg_loss, avg_ast_loss, avg_ret_loss, avg_fp_loss, avg_kl_loss = 0., 0., 0., 0., 0.
            dropout_dict = {model.encoder.drop_prob: config.drop_rate}

            for b in range(config.num_batches):
                nodes, edges, targets, \
                        ret_type, fp_type, fp_type_targets,\
                        _ = loader.next_batch()
                feed_dict = dropout_dict
                feed_dict.update({model.nodes: nodes, model.edges: edges, model.targets: targets})
                feed_dict.update({model.return_type: ret_type})
                feed_dict.update({model.formal_params: fp_type, model.formal_param_targets: fp_type_targets})
                # run the optimizer
                loss, ast_loss, ret_loss, fp_loss, kl_loss, _ = \
                    sess.run([model.loss, model.ast_gen_loss,
                              model.ret_gen_loss, model.fp_gen_loss,
                              model.KL_loss, model.train_op], feed_dict=feed_dict)

                avg_loss += np.mean(loss)
                avg_ast_loss += np.mean(ast_loss)
                avg_ret_loss += np.mean(ret_loss)
                avg_fp_loss += np.mean(fp_loss)
                avg_kl_loss += np.mean(kl_loss)

                step = i * config.num_batches + b
                if step % config.print_step == 0:
                    logger.info('{}/{} (epoch {}) '
                                'loss: {:.3f}, gen loss: {:.3f}, '
                                'ret loss: {:.3f}, fp loss: {:.3f}, '
                                'KL loss: {:.3f}. '
                                .format(step,
                                        config.num_epochs * config.num_batches,
                                        i + 1, avg_loss / (b + 1), avg_ast_loss / (b + 1),
                                        avg_ret_loss / (b + 1), avg_fp_loss / (b + 1),
                                        avg_kl_loss / (b + 1)))
                    message = ('{}/{} (epoch {}) '
                                'loss: {:.3f}, gen loss: {:.3f}, '
                                'ret loss: {:.3f}, fp loss: {:.3f}, '
                                'KL loss: {:.3f}. \n'
                                .format(step,
                                        config.num_epochs * config.num_batches,
                                        i + 1, avg_loss / (b + 1), avg_ast_loss / (b + 1),
                                        avg_ret_loss / (b + 1), avg_fp_loss / (b + 1),
                                        avg_kl_loss / (b + 1)))
                    analysis_file.write(message)
                    analysis_file.flush()
                    os.fsync(analysis_file.fileno())



            if (i + 1) % config.checkpoint_step == 0:
                checkpoint_dir = os.path.join(clargs.save, 'model{}.ckpt'.format(i + 1))
                saver.save(sess, checkpoint_dir)
                with open(os.path.join(clargs.save + '/config.json'), 'w') as f:
                    json.dump(dump_config(config), fp=f, indent=2)
                with open(os.path.join(clargs.save + '/pid.json'), 'w') as f:
                    json.dump({'pid': os.getpid()}, fp=f, indent=2)
                logger.info('Model checkpoint: {}. Average for epoch , '
                            'loss: {:.3f}'.format
                            (checkpoint_dir, avg_loss / config.num_batches))
                message = 'Model checkpoint: {}. Average for epoch , loss: {:.3f}\n'.format(checkpoint_dir, avg_loss / config.num_batches)
                analysis_file.write(message)
                analysis_file.flush()
                os.fsync(analysis_file.fileno())


# %%
if __name__ == '__main__':
    folder_name = 'testing-600'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='save/' + folder_name + "/",
                        help='checkpoint model during training here')
    parser.add_argument('--data', type=str, default='../data_extractor/data/' + folder_name + "/",
                        help='load data from here')
    parser.add_argument('--config', type=str, default='config.json',
                        help='config file (see description above for help)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--topK', type=int, default=10,
                        help='plot only the top-k labels')
    parser.add_argument('--filename', type=str, help='name of data file and dir name')
    clargs_ = parser.parse_args()
    if not os.path.exists(clargs_.save):
        os.makedirs(clargs_.save)
    if not os.path.exists(clargs_.save + '/plots'):
        os.makedirs(clargs_.save + '/plots')
    clargs_.folder_name = folder_name
    sys.setrecursionlimit(clargs_.python_recursion_limit)
    if clargs_.config and clargs_.continue_from:
        parser.error('Do not provide --config if you are continuing from checkpointed model')
    if not clargs_.config and not clargs_.continue_from:
        parser.error('Provide at least one option: --config or --continue_from')
    train(clargs_)

    if clargs_.continue_from is None:
        clargs_.continue_from = clargs_.save

    clargs_.filename = 'plot_' + folder_name + '.png'
    plot(clargs_)
