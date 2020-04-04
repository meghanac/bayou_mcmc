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
import tensorflow as tf

import logging
import argparse
import sys
import json
import textwrap

from data_extractor.data_loader import Loader
from trainer_gan.model import Model
from trainer_gan.utils import read_config, dump_config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

    loader = Loader(clargs, config)
    model = Model(config)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(os.path.join(clargs.save, 'loss_values.log'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Process id is {}'.format(os.getpid()))
    logger.info('GPU device is {}\n\t'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info('Amount of data used is {}\n\t'.format(config.num_batches * config.batch_size))

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
            avg_disc_loss, avg_gen_loss = 0., 0.

            for b in range(config.num_batches):

                nodes, edges, _, _, _, _, keywords = loader.next_batch()
                feed_dict = {}
                feed_dict.update({model.nodes: nodes})
                feed_dict.update({model.edges: edges})
                feed_dict.update({model.keywords: keywords})

                # for j in range(10):
                gen_loss, fake_logits2, _ = \
                    sess.run([model.gen_loss, model.faker.logits,
                              model.gen_train_op], feed_dict=feed_dict)

                # run the optimizer
                disc_loss, fake_logits1, real_logits1, _ = \
                    sess.run([model.disc_loss, model.faker.logits, model.real.logits,
                              model.disc_train_op], feed_dict=feed_dict)



                # fake_logits1 += np.mean(fake_logits1)
                # real_logits1 += np.mean(real_logits1)
                # fake_logits2 += np.mean(fake_logits2)

                avg_disc_loss += np.mean(disc_loss)
                avg_gen_loss += np.mean(gen_loss)

                step = i * config.num_batches + b
                if step % config.print_step == 0:
                    logger.info('{}/{} (epoch {}) '
                          'gen_loss: {:.3f}, disc_loss {:.3f}.'.format(step,
                                                                            config.num_epochs * config.num_batches,
                                                                            i + 1, avg_gen_loss / (b + 1),
                                                                            avg_disc_loss / (b + 1)))
                    # print('Discriminator fake logits :: {:.3f}, real logits :: {:.3f}, \t'.format((fake_logits1), (real_logits1)))
                    # print('Generator fake logits :: {:.3f}, \n'.format((fake_logits2)))

            if (i + 1) % config.checkpoint_step == 0:
                checkpoint_dir = os.path.join(clargs.save, 'model{}.ckpt'.format(i+1))
                saver.save(sess, checkpoint_dir)
                with open(os.path.join(clargs.save + '/config.json'), 'w') as f:
                    json.dump(dump_config(config), fp=f, indent=2)
                with open(os.path.join(clargs.save + '/pid.json'), 'w') as f:
                    json.dump({'pid': os.getpid()}, fp=f, indent=2)
                logger.info('Model checkpointed: {}. Average for epoch , '
                      'loss: {:.3f}'.format
                      (checkpoint_dir, (avg_disc_loss + avg_gen_loss) / config.num_batches))


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='save',
                        help='checkpoint model during training here')
    parser.add_argument('--data', type=str, default='../data_extractor/data',
                        help='load data from here')
    parser.add_argument('--config', type=str, default=None,
                        help='config file (see description above for help)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='ignore config options and continue training model checkpointed here')
    clargs_ = parser.parse_args()
    if not os.path.exists(clargs_.save):
        os.makedirs(clargs_.save)
    sys.setrecursionlimit(clargs_.python_recursion_limit)
    if clargs_.config and clargs_.continue_from:
        parser.error('Do not provide --config if you are continuing from checkpointed model')
    if not clargs_.config and not clargs_.continue_from:
        parser.error('Provide at least one option: --config or --continue_from')
    train(clargs_)
