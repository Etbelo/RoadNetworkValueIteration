
#include "MDPValueIteration.h"

#include <algorithm>  // sdt::max_element
#include <cmath>      // std::floor, std::max, std::min, std::exp, std::abs
#include <iostream>   // std::cout, std::endl

#include "omp.h"

namespace backend {

auto evaluate_mdp(int *pi_data, float *J_data, int *P_indptr, int *P_indices,
                  float *P_data, const int P_nnz, const int num_nodes,
                  const int max_actions, const int num_charges,
                  const float alpha, const float error_min,
                  const int num_blocks) -> void {
#ifdef VERBOSE
    std::cout << std::endl << "[cpp backend] running" << std::endl;
#endif

    const auto num_states = num_nodes * num_nodes * num_charges;

    // Construct T matrix from existing memory
    Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> P(
        num_states * max_actions, num_states, P_nnz, P_indptr, P_indices,
        P_data);

    // Construct result vectors from existing memory
    Eigen::Map<Eigen::VectorXi> pi(pi_data, num_states);
    Eigen::Map<Eigen::VectorXf> J(J_data, num_states);

    // Evaluate MDP using asynchronous value iteration
    async_value_iteration(P, pi, J, num_states, num_nodes, max_actions, alpha,
                          error_min, num_blocks);
}

auto async_value_iteration(
    Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> P,
    Eigen::Ref<Eigen::VectorXi> pi, Eigen::Ref<Eigen::VectorXf> J,
    const int num_states, const int num_nodes, const int max_actions,
    const float alpha, const float error_min, const int num_blocks) -> void {
    // Split statespace into blocks
    // => Computation load dsitribution: num_blocks can be higher than
    // num_threads Schedule decides over block order
    // => Each thread handles dynamic number of blocks per iteration

    const auto block_size = num_states / num_blocks;
    const auto num_nodes_sq = num_nodes * num_nodes;
    auto global_error = -1.0f;
    std::vector<float> local_errors(omp_get_max_threads(), -1.0f);

    do {
#pragma omp parallel for schedule(guided, 1)
        for (int i = 0; i < num_blocks; ++i) {
            // State bounds for current block
            const auto low_bound = i * block_size;
            const auto up_bound =
                i == num_blocks - 1 ? num_states : (i + 1) * block_size;

            // Handle current block
            const auto block_error =
                update_block(low_bound, up_bound, P, J, pi, max_actions, alpha,
                             num_nodes, num_nodes_sq);

            // Update thread-local error with error of current block
            local_errors.at(omp_get_thread_num()) =
                std::max(local_errors.at(omp_get_thread_num()), block_error);
        }

        // Evaluate global error after all threads have finished
        global_error = -1.0f;

        for (auto &local_error : local_errors) {
            // Update global_error
            if (local_error > global_error) {
                global_error = local_error;
            }

            // Reset local_error for next iteration
            local_error = -1.0f;
        }

#ifdef VERBOSE
        std::cout << "[cpp backend] global_error: " << global_error
                  << std::endl;
#endif

    } while (global_error > error_min);
}

auto update_block(const unsigned int low_bound, const unsigned int up_bound,
                  Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> P,
                  Eigen::Ref<Eigen::VectorXf> J, Eigen::Ref<Eigen::VectorXi> pi,
                  const int max_actions, const float alpha, const int num_nodes,
                  const unsigned int num_nodes_sq) -> float {
    float local_error = -1.0;

    // Loop over all states in current block
    for (auto state = low_bound; state < up_bound; ++state) {
        float J_temp = 0.0;
        int pi_temp = 0;

        // Retrieve state information
        int charge, tar_node, cur_node;
        std::tie(charge, tar_node, cur_node) =
            decode_state(state, num_nodes, num_nodes_sq);

        // Loop over all actions
        for (int action = 0; action < max_actions; ++action) {
            // Total sum of costs for current action
            float J_cur = 0.0;

            // Calculate current stage cost
            const auto g = stage_cost(charge, tar_node, cur_node, action);
            const auto row_in_P = state * max_actions + action;
            float row_sum = 0.0;

            // Loop over non zero successor states to update action cost J_cur
            for (Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>>::
                     InnerIterator it(P, row_in_P);
                 it; ++it) {
                row_sum += it.value();

                // Update action cost with successor state it.col()
                J_cur += it.value() * (g + alpha * J(it.col()));
            }

            // Consider state if action is valid and total cost is smaller
            if (std::abs(row_sum - 1.0) < 0.1 &&
                (J_cur < J_temp || J_temp == 0.0)) {
                J_temp = J_cur;
                pi_temp = action;
            }
        }

        // Update block-local error
        local_error = std::max(local_error, std::abs(J(state) - J_temp));

        // Update state cost and policy
        J(state) = J_temp;
        pi(state) = pi_temp;
    }

    return local_error;
}

auto decode_state(int state, int num_nodes, int num_nodes_sq)
    -> std::tuple<int, int, int> {
    const auto charge = state / num_nodes_sq;
    const auto tar_node = state % num_nodes_sq / num_nodes;
    const auto cur_node = state % num_nodes_sq % num_nodes;

    return {charge, tar_node, cur_node};
}

auto stage_cost(int charge, int tar_node, int cur_node, int action) -> float {
    // Reward arriving at target
    if (cur_node == tar_node && action == 0) {
        return -100.0;
    }

    // Punish staying at current node without action
    if (cur_node != tar_node && action == 0) {
        return 20.0;
    }

    // Punish having no charge left
    if (charge == 0) {
        return 100.0;
    }

    // Punish Moving too much
    if (action > 0) {
        return 5.0;
    }

    return 0.0;
}

}  // namespace backend
