import math

from mcmc import MCMCProgram
import numpy as np

ALL_DATA_1K_05_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'

"""
Reference: 
https://stats.stackexchange.com/questions/99375/gelman-and-rubin-convergence-diagnostic-how-to-generalise-to-work-with-vectors

"""


def run_gelman_rubin(data_path=ALL_DATA_1K_05_MODEL_PATH, num_chains=3, num_iterations=2):
    average_states = []
    mcmc_progs = []
    mean_sq_dist = []

    for i in range(num_chains):
        prog = MCMCProgram(data_path, debug=False, verbose=False, save_states=True)
        constraints = ['java.io.FileInputStream.read(byte[])', 'java.nio.ByteBuffer.getInt(int)',
                       'java.lang.String.format(java.lang.String,java.lang.Object[])']
        return_type = ['void']
        formal_params = ['DSubTree', 'String']

        prog.init_program(constraints, return_type, formal_params, ordered=True)

        for _ in range(num_iterations):
            prog.mcmc()

        avg_state = np.zeros_like(prog.states[0][0])
        for state in prog.states:
            avg_state += state[0]/num_iterations

        sq_dists = []
        for state in prog.states:
            sq_dists.append(np.matmul(state[0] - avg_state, (state[0] - avg_state).T))

        mean_sq_dist.append(1/(num_iterations-1) * sum(sq_dists)[0])

        average_states.append(avg_state)
        mcmc_progs.append(prog)

    avg_state = np.zeros_like(average_states[0])
    for state in average_states:
        avg_state += state/num_chains

    chain_sq_dists = []
    for state in average_states:
        chain_sq_dists.append(np.matmul(state - avg_state, (state - avg_state).T))

    B = num_iterations/(num_chains - 1) * sum(chain_sq_dists)[0]

    W = 1/num_chains * sum(mean_sq_dist)[0]

    Var = (1 - 1/num_iterations) * W + 1/num_iterations * B

    R = math.sqrt(Var/W)

    print(R)

    # eigenvalues, _ = np.linalg.eig(1/num_iterations * np.matmul(np.linalg.inv(W), Var))
    # max_eig = max(eigenvalues)
    #
    # print(max_eig)
    #
    # assert max_eig >= 0, "Max eigenvalue must be positive"
    #
    # R = (num_iterations - 1)/num_iterations + (num_chains + 1)/num_chains * max_eig
    #
    # print("R value:", R)


run_gelman_rubin()
