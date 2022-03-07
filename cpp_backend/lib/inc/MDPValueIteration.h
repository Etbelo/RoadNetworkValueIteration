#ifndef __MDP_VALUE_ITERATION_H__
#define __MDP_VALUE_ITERATION_H__

#include <Eigen/Dense>   // Eigen::Map, Eigen::MatrixXf, Eigen::VectorXf
#include <Eigen/Sparse>  // Eigen::SparseMatrix
#include <iostream>      // std::cout, std::endl
#include <tuple>         // std::tie, std::tuple
#include <vector>        // std::vector

namespace backend {

/** \addtogroup MDPValueIteration
 *  @{
 */

/**
 * @brief Acquire a policy from given Markov Decision Process and its
 * parameters in numpy format.
 *
 * @param[in,out] pi_data Pointer to output policy array (np vector)
 * @param[in,out] J_data Pointer to output cost array (np vector)
 * @param[in,out] P_indptr Pointer to indptr array of P matrix (np csr)
 * @param[in,out] P_indices Pointer to indices array of P matrix (np csr)
 * @param[in,out] P_data Pointer to data array of P matrix (np csr)
 * @param[in] P_nnz Number of non-zero elements in P matrix (np csr)
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] max_actions Maximum number of actions in mdp
 * @param[in] num_charges Maximum number of possible charges in mdp
 * @param[in] alpha Discount factor for value iteration
 * @param[in] error_min Minimum error as target for greedy algorithm
 * @param[in] num_blocks Number of statespace blocks for threads to work on
 */
auto evaluate_mdp(int *pi_data, float *J_data, int *P_indptr, int *P_indices,
                  float *P_data, int P_nnz, int num_nodes, int max_actions,
                  int num_charges, float alpha, float error_min, int num_blocks)
    -> void;

/**
 * @brief Acquire a policy from probability matrix in Eigen format. Use all
 * available threads of current CPU to evaluate policy using asynchronous value
 * iteration algorithm.
 *
 * @param[in,out] P Reference to probability matrix (Eigen csr)
 * @param[in,out] pi Reference to output policy array (Eigen vector)
 * @param[in,out] J Reference to output cost array (Eigen vector)
 * @param[in] num_states Total number of states in mdp
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] max_actions Maximum number of actions in mdp
 * @param[in] alpha Discount factor for value iteration
 * @param[in] error_min Minimum error as target for greedy algorithm
 * @param[in] num_blocks Number of statespace blocks for threads to work on
 */
auto async_value_iteration(
    Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> P,
    Eigen::Ref<Eigen::VectorXi> pi, Eigen::Ref<Eigen::VectorXf> J,
    int num_states, int num_nodes, int max_actions, float alpha,
    float error_min, int num_blocks) -> void;

/**
 * @brief Update statespace block in single loop of value iteration.
 *
 * @param[in] state_low Low bound of current state block
 * @param[in] state_high Upper bounnd of current state block
 * @param[in,out] P Reference to P matrix (Eigen csr)
 * @param[in,out] J Reference to cost array (Eigen vector)
 * @param[in,out] pi Referenc to policy array (Eigen vector)
 * @param[in] max_actions Maximum number of actions in mdp
 * @param[in] alpha Discount factor for value iteration
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] num_nodes_sq Total number of nodes squared in mdp
 * @return float Local cost error in current state block
 */
auto update_block(int state_low, int state_high,
                  Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> P,
                  Eigen::Ref<Eigen::VectorXf> J, Eigen::Ref<Eigen::VectorXi> pi,
                  int max_actions, float alpha, int num_nodes, int num_nodes_sq)
    -> float;

/**
 * @brief Decode state from number to tuple (cur_charge, tar_node, cur_node).
 *
 * @param[in] state State to decode from
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] num_nodes_sq Total number of nodes squared in mdp
 * @return std::tuple<int, int, int> State tuple (cur_charge, tar_node,
 * cur_node)
 */
auto decode_state(int state, int num_nodes, int num_nodes_sq)
    -> std::tuple<int, int, int>;

/**
 * @brief Get stage cost given the state and taken action.
 *
 * @param[in] cur_charge Current charge in state
 * @param[in] tar_node Current target node in state
 * @param[in] cur_node Current node in state
 * @param[in] action Taken action
 * @return float Stage cost
 */
auto stage_cost(int cur_charge, int tar_node, int cur_node, int action)
    -> float;

/** @}*/

}  // namespace backend

#endif  // __MDP_VALUE_ITERATION_H__
