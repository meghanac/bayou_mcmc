from experiments import Experiments
from data_extractor.dataset_creator import IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW
from data_extractor.graph_analyzer import ALL_DATA_NO_DUP


num_iterations = 3
data_dir_name = 'new_all_data_1k_vocab_no_duplicates'
model_dir_path = '../trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'

exp_dir_name = "testing"

exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=False, train_test_set_dir_name='/fixed_train_test_sets/')

exp.run_mcmc(EX_CS, NEW, save_run=False, num_test_progs=20, verbose=False)