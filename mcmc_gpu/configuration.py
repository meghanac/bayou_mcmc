import os
import json
from trainer_vae.utils import read_config
from numba import jit, cuda

MAX_LOOP_NUM = 3
MAX_BRANCHING_NUM = 3
MAX_AST_DEPTH = 32
TEMP = 'temp'

class Configuration:
    def __init__(self, save_dir):
        # Initialize model
        self.save_dir = save_dir
        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file) as f:
            self.config_obj = read_config(json.load(f), infer=True)

        # Initialize model configurations
        self.max_num_api = self.config_obj.max_ast_depth
        self.max_length = MAX_AST_DEPTH
        self.config_obj.max_ast_depth = 1
        self.config_obj.max_fp_depth = 1
        self.config_obj.batch_size = 1

        self.batch_size = self.config_obj.batch_size
        self.max_ast_depth = self.config_obj.max_ast_depth
        self.latent_size = self.config_obj.latent_size
        self.decoder = self.config_obj.decoder

        # Initialize conversion dictionaries
        self.vocab = self.config_obj.vocab
        self.vocab_size = self.config_obj.vocab.api_dict_size
        self.vocab2node = self.config_obj.vocab.api_dict
        self.vocab2node[TEMP] = -1
        self.node2vocab = dict(zip(self.vocab2node.values(), self.vocab2node.keys()))
        self.rettype2num = self.config_obj.vocab.ret_dict
        self.num2rettype = dict(zip(self.rettype2num.values(), self.rettype2num.keys()))
        self.fp2num = self.config_obj.vocab.fp_dict
        self.num2fp = dict(zip(self.fp2num.values(), self.fp2num.keys()))