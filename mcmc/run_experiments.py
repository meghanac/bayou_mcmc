from experiments import Experiments
from data_extractor.dataset_creator import IN_CS, IN_API, EX_API, EX_CS, SEEN, NEW
from data_extractor.graph_analyzer import ALL_DATA_NO_DUP


num_iterations = 10
data_dir_name = ALL_DATA_NO_DUP
model_dir_path = '../trainer_vae/save/all_training_data_1.38m_large_config'

exp_dir_name = "testing"

exp = Experiments(data_dir_name, model_dir_path, exp_dir_name, num_iterations, save_mcmc_progs=True)

exp.run_mcmc(IN_CS, NEW)