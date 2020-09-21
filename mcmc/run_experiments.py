from experiments import Experiments
from data_extractor.dataset_creator import IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW
from data_extractor.graph_analyzer import ALL_DATA_NO_DUP


num_iterations = 3
data_dir_name = 'new_all_data_1k_vocab_no_duplicates'
model_dir_path = '../trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'

exp_dir_name = "testing-100"

exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=True,
                  train_test_set_dir_name='/novel_min_2/')

exp.run_mcmc(IN_CS, NEW, save_run=True, num_test_progs=5, verbose=False)
print("\n\n\n\n\n\n")
exp.run_mcmc(IN_API, NEW, save_run=True, num_test_progs=5, verbose=False)
print("\n\n\n\n\n\n")

# num_iterations = 200
# exp_dir_name = "testing-200"
#
# exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=False,
#                   train_test_set_dir_name='/train_test_sets_new_ex/')
#
# exp.run_mcmc(EX_CS, NEW, save_run=True, num_test_progs=5, verbose=False)
# exp.run_mcmc(IN_API, NEW, save_run=True, num_test_progs=5, verbose=False)
#
# print("\n\n\n\n\n\n")
#
# num_iterations = 330
# exp_dir_name = "testing-330"
#
# exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=False,
#                   train_test_set_dir_name='/train_test_sets_new_ex/')
#
# # exp.run_mcmc(EX_CS, NEW, save_run=True, num_test_progs=5, verbose=False)
# exp.run_mcmc(IN_CS, NEW, save_run=True, num_test_progs=5, verbose=False)
# exp.run_mcmc(IN_API, NEW, save_run=True, num_test_progs=5, verbose=False)

