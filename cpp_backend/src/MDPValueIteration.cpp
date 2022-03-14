
#include "MDPValueIteration.h"

#include <algorithm>  // std::max, std::min,
#include <cmath>      // std::floor, std::abs
#include <iostream>   // std::cout, std::endl

#include "omp.h"

namespace cpp_backend {

auto EvaluateMdp(int *pi_data, float *J_data, int *P_indptr, int *P_indices,
                 float *P_data, const int P_nnz, const int num_nodes,
                 const int max_actions, const int num_charges,
                 const float alpha, const float error_min, const int num_blocks)
    -> void {
#ifdef VERBOSE
    std::cout << std::endl << "[cpp backend] (INFO) running" << std::endl;
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
    AsyncValueIteration(P, pi, J, num_states, num_nodes, max_actions, alpha,
                        error_min, num_blocks);
}

auto AsyncValueIteration(
    const Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> &P,
    Eigen::Ref<Eigen::VectorXi> pi, Eigen::Ref<Eigen::VectorXf> J,
    const int num_states, const int num_nodes, const int max_actions,
    const float alpha, const float error_min, const int num_blocks) -> void {
    // Split statespace into blocks
    // => Computation load dsitribution: blocks >= threads
    // Schedule decides over block order
    // => Each thread handles dynamic number of blocks per iteration

    const auto threads = omp_get_max_threads();
    const auto blocks = std::max(threads, num_blocks);

    const auto block_size = num_states / blocks;
    const auto num_nodes_sq = num_nodes * num_nodes;
    auto global_error = -1.0f;
    std::vector<float> local_errors(threads, -1.0f);

    do {
#pragma omp parallel for schedule(guided, 1)
        for (int i = 0; i < blocks; ++i) {
            // State bounds for current block
            const auto state_low = i * block_size;
            const auto state_high =
                i == blocks - 1 ? num_states : (i + 1) * block_size;

            // Handle current block
            const auto block_error =
                UpdateBlock(state_low, state_high, P, J, pi, max_actions, alpha,
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
        std::cout << "[cpp backend] (INFO) global_error: " << global_error
                  << std::endl;
#endif

    } while (global_error > error_min);
}

auto UpdateBlock(
    const int state_low, const int state_high,
    const Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> &P,
    Eigen::Ref<Eigen::VectorXf> J, Eigen::Ref<Eigen::VectorXi> pi,
    const int max_actions, const float alpha, const int num_nodes,
    const int num_nodes_sq) -> float {
    // Block-local error
    float block_error = -1.0;

    // Loop over all states in current block
    for (auto state = state_low; state < state_high; ++state) {
        float J_temp = 0.0;
        int pi_temp = 0;

        // Retrieve state information
        const auto state_tuple =
            get_state_tuple(state, num_nodes, num_nodes_sq);

        // Loop over all actions
        for (int action = 0; action < max_actions; ++action) {
            // Total sum of costs for current action
            float J_cur = 0.0;

            // Calculate current stage cost
            const auto g = get_stage_cost(state_tuple, action);

            // Loop over non zero successor states to update action cost J_cur
            const auto p_row = state * max_actions + action;
            float p_row_sum = 0.0;

            for (Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>>::
                     InnerIterator it(P, p_row);
                 it; ++it) {
                // Update total row sum
                p_row_sum += it.value();

                // Update action cost with successor state it.col()
                J_cur += it.value() * (g + alpha * J[it.col()]);
            }

            // Consider state if action is valid and total cost is smaller
            if (std::abs(p_row_sum - 1.0) < 0.1 &&
                (J_cur < J_temp || J_temp == 0.0)) {
                J_temp = J_cur;
                pi_temp = action;
            }
        }

        // Update block-local error
        block_error = std::max(block_error, std::abs(J[state] - J_temp));

        // Update state cost and policy
        J(state) = J_temp;
        pi(state) = pi_temp;
    }

    return block_error;
}

auto get_state_tuple(const int state, const int num_nodes,
                     const int num_nodes_sq) -> StateTuple {
    StateTuple state_tuple;

    state_tuple.cur_charge = state / num_nodes_sq;
    state_tuple.tar_node = state % num_nodes_sq / num_nodes;
    state_tuple.cur_node = state % num_nodes_sq % num_nodes;

    return state_tuple;
}

auto get_stage_cost(StateTuple state_tuple, const int action) -> float {
    // Reward arriving at target
    if (state_tuple.cur_node == state_tuple.tar_node && action == 0) {
        return -100.0;
    }

    // Handle cost if not arrived at target yet
    float t_stage_cost = 0.0;

    if (state_tuple.cur_charge == 0) {
        // Punish having no charge left
        t_stage_cost += 100;
    } else {
        //  Reward aquiring charge
        t_stage_cost -= 50.0;
    }

    if (action > 0) {
        // Punish moving more than needed
        t_stage_cost += 10.0;
    }

    return t_stage_cost;
}

}  // namespace cpp_backend
