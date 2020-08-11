import math

from mcmc import MCMCProgram
import numpy as np

from test_utils import print_summary_logs

ALL_DATA_1K_05_MODEL_PATH = '/Users/meghanachilukuri/bayou_mcmc/trainer_vae/save/all_data_1k_vocab_0.5_KL_beta'

"""
Reference: 
Original:
https://stats.stackexchange.com/questions/99375/gelman-and-rubin-convergence-diagnostic-how-to-generalise-to-work-with-vectors

Improved:
https://arxiv.org/pdf/1812.09384.pdf
"""


def run_gelman_rubin(data_path=ALL_DATA_1K_05_MODEL_PATH, num_chains=10, num_iterations=60, verbose=False):
    # Checks
    assert num_iterations % 3 == 0

    cube_root = math.floor(num_iterations ** (1 / 3))
    sq_root = math.ceil(num_iterations ** (1 / 2))

    b = 0  # batch size
    for i in range(cube_root, sq_root):
        if i % 3 == 0 and num_iterations % i == 0:
            b = i

    assert b != 0

    # Original PSRF
    average_states = []
    mcmc_progs = []
    mean_sq_dist = []

    for i in range(num_chains):
        prog = MCMCProgram(data_path, debug=False, verbose=False, save_states=True)
        constraints = ['java.util.ArrayList<Tau_E>.ArrayList()']
        return_type = ['void']
        formal_params = ['DSubTree', 'String']

        prog.init_program(constraints, return_type, formal_params, ordered=True)

        for _ in range(num_iterations):
            prog.mcmc()

        if verbose:
            print_summary_logs(prog)

        avg_state = np.zeros_like(prog.states[0][0])
        for state in prog.states:
            avg_state += state[0][0]/num_iterations

        sq_dists = []
        for state in prog.states:
            sq_dists.append(np.matmul(state[0][0] - avg_state, (state[0][0] - avg_state).T))

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

    print("Original Gelman-Rubin PSRF:", R)

    # multivariate approach
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

    # Improved PSRF
    a = num_iterations / b

    def get_sq_dist_per_chain(b):
        sq_dist_per_chain = []
        for chain in mcmc_progs:
            mean_per_batch = []
            state = chain.states[0][0]
            for n in range(1, num_iterations):
                if n % b == 0:
                    mean_per_batch.append(1 / b * state)
                    state = np.zeros_like(chain.states[0][0])
                state += chain.states[i][0]
            mean_per_batch.append(1 / b * state)
            assert len(mean_per_batch) == a, str(a) + " len: " + str(len(mean_per_batch))

            mean_per_batch = [np.matmul(y - avg_state, (y - avg_state).T) for y in mean_per_batch]
            sq_dist_per_chain.append(sum(mean_per_batch)[0])

        return sq_dist_per_chain

    sq_dist_per_chain = get_sq_dist_per_chain(b)
    tau_b_sq = b / (a * num_chains - 1) * sum(sq_dist_per_chain)[0]

    b /= 3
    a = num_iterations / b
    sq_dist_per_chain = get_sq_dist_per_chain(b)
    tau_b_third_sq = b / (a * num_chains - 1) * sum(sq_dist_per_chain)[0]

    tau_L_sq = 2 * tau_b_sq - tau_b_third_sq

    sigma_l_sq = (num_iterations - 1) / num_iterations * W + tau_L_sq / num_iterations

    R_L = math.sqrt(sigma_l_sq / W)

    print("Improved PSRF:", R_L)


run_gelman_rubin()


# low freq space
# num_chains=10, num_iterations=20, R = 1.01379928817122
# num_chains=5, num_iterations=30, R = 1.013167297967596

# array list
# num_chains=5, num_iterations=30, R = 1.006
# num_chains=5, num_iterations=100, R=1.01152779091537