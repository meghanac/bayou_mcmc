import unittest

from data_extractor.dataset_creator import IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW, MIN_EQ, MAX_EQ
from experiments import Experiments

MIN_LEN = 'min_length'
MAX_LEN = 'max_length'


class TestExperiments(unittest.TestCase):
    def get_experiments_class(self, label, verbose=False):
        num_iterations = 50
        data_dir_name = 'new_all_data_1k_vocab_no_duplicates'
        model_dir_path = '../trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'
        # model_dir_path = '../trainer_vae/save/final_novel_1k_min_2_smallest_config'

        exp_dir_name = "tests"

        if label == SEEN:
            train_test_dir_name = '/seen_min_2/'
        else:
            train_test_dir_name = '/novel_min_2/'

        # api_proposal_probs = {INSERT: 0.2, DELETE: 0.3, SWAP: 0.1, REPLACE: 0.2, ADD_DNODE: 0.0, GROW_CONST: 0.2,
        #                       GROW_CONST_UP: 0.0}
        # cs_proposal_probs = {INSERT: 0.15, DELETE: 0.3, SWAP: 0.2, REPLACE: 0.25, ADD_DNODE: 0.0, GROW_CONST: 0.1,
        #                      GROW_CONST_UP: 0.0}

        exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=False,
                          # train_test_set_dir_name='/train_test_sets_min_2/')
                          train_test_set_dir_name=train_test_dir_name, verbose=verbose)
        return exp

    def test_load_post_dicts(self):

        dir_path = 'experiments/'

        novel_exp = self.get_experiments_class(NEW)
        for category in [MIN_EQ, MAX_EQ, IN_API, EX_API, IN_CS, EX_CS]:
            path = dir_path + category + "/" + NEW + "/"
            for i in range(1, 9):
                novel_exp.load_posterior_distribution_from_file(path + str(i) + "/", category, NEW)

        for category in [MIN_EQ, MAX_EQ, IN_API, EX_API, IN_CS, EX_CS]:
            for i in range(1, 4):
                path = "experiments_aws/"
                if category == MIN_EQ:
                    path += MIN_LEN
                elif category == MAX_EQ:
                    path += MAX_LEN
                else:
                    path += category
                if i > 1:
                    path += str(i)
                path += "/"
                novel_exp.load_posterior_distribution_from_file(path, category, NEW)

        accu_exp = self.get_experiments_class(SEEN)
        for category in [MIN_EQ, MAX_EQ, IN_API, EX_API, IN_CS, EX_CS]:
            path = dir_path + category + "/" + SEEN + "/"
            for i in range(1, 9):
                accu_exp.load_posterior_distribution_from_file(path + str(i) + "/", category, SEEN)

        for category in [MIN_EQ, MAX_EQ, IN_API, EX_API, IN_CS, EX_CS]:
                novel_exp.calculate_metrics(category, NEW)
                accu_exp.calculate_metrics(category, SEEN)

    def test_single(self):
        category = MIN_EQ
        label = NEW
        novel_exp = self.get_experiments_class(label)
        path = 'experiments/' + category + "/" + label + "/1/"
        novel_exp.load_posterior_distribution_from_file(path, category, label)

        for i in novel_exp.posterior_dists[category][label]:
            print(i)
            print(novel_exp.posterior_dists[category][label][i])
            print("")

        novel_exp.calculate_metrics(category, NEW)


if __name__ == '__main__':
    unittest.main()
