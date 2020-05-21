import argparse
import math
import os
import sys
import textwrap
from copy import deepcopy
import numpy as np
import random
import json
import tensorflow as tf

from mcmc.node import Node, SIBLING_EDGE, CHILD_EDGE, DNODES, DBRANCH, DLOOP, DEXCEPT, START, STOP, EMPTY
from mcmc.proposals.insertion_proposals import ProposalWithInsertion


class AddDnodeProposal(ProposalWithInsertion):

    def __init__(self, tree_modifier, decoder, top_k_prob=0.95):
        super().__init__(tree_modifier, decoder, top_k_prob)

    def add_random_dnode(self, curr_prog, initial_state):
        """
        Adds a DBranch, DLoop or DExcept to a random node in the current program.
        :return: (Node) the dnode node
        """
        # Temporarily save curr_prog and initial_state
        self.curr_prog = curr_prog
        self.initial_state = initial_state

        dnode_type = random.choice([DBRANCH, DLOOP, DEXCEPT])

        parent, _ = self._get_valid_random_node(curr_prog)

        if parent is None:
            return None

        assert parent.child is None or parent.parent.api_name == DBRANCH, \
            "WARNING: there's a bug in get_valid_random_node because parent node has child"

        # Grow dnode type
        if dnode_type == DBRANCH:
            dnode, ln_prob = self._grow_dbranch(parent)
        elif dnode_type == DLOOP:
            dnode, ln_prob = self._grow_dloop_or_dexcept(parent, True)
        else:
            dnode, ln_prob = self._grow_dloop_or_dexcept(parent, False)

        # Reset self.curr_prog and self.initial_state
        self.curr_prog = None
        self.initial_state = None

        return curr_prog, dnode, ln_prob

    def undo_add_random_dnode(self, dnode):
        """
        Removes the dnode that was added in add_random_dnode().
        :param dnode: (Node) dnode that is to be removed
        :return:
        """
        dnode_sibling = dnode.sibling
        dnode_parent = dnode.parent
        dnode_parent.remove_node(SIBLING_EDGE)
        dnode_parent.add_node(dnode_sibling, SIBLING_EDGE)