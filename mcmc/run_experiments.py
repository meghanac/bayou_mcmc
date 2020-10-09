from experiments import Experiments
from data_extractor.dataset_creator import MIN_EQ, MAX_EQ, IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW
from data_extractor.graph_analyzer import ALL_DATA_NO_DUP
from mcmc import INSERT, DELETE, SWAP, REPLACE, ADD_DNODE, GROW_CONST_UP, GROW_CONST

num_iterations = 50
data_dir_name = 'new_all_data_1k_vocab_no_duplicates'
model_dir_path = '../trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'
# model_dir_path = '../trainer_vae/save/final_novel_1k_min_2_smallest_config'

exp_dir_name = "testing-200"

api_proposal_probs = {INSERT: 0.2, DELETE: 0.3, SWAP: 0.1, REPLACE: 0.2, ADD_DNODE: 0.0, GROW_CONST: 0.2,
                      GROW_CONST_UP: 0.0}
cs_proposal_probs = {INSERT: 0.15, DELETE: 0.3, SWAP: 0.2, REPLACE: 0.25, ADD_DNODE: 0.0, GROW_CONST: 0.1,
                     GROW_CONST_UP: 0.0}

exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=True,
                  # train_test_set_dir_name='/train_test_sets_min_2/')
                  train_test_set_dir_name='/novel_min_2/')

exp.run_mcmc(MAX_EQ, NEW, save_run=True, num_test_progs=5, verbose=True, save_step=1, proposal_probs=api_proposal_probs)
print("\n\n\n\n\n\n")


# exp.run_mcmc(IN_API, NEW, save_run=True, num_test_progs=5, verbose=False, save_step=1)
# print("\n\n\n\n\n\n")

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

