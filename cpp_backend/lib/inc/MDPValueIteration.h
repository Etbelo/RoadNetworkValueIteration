#ifndef __MDP_VALUE_ITERATION_H__
#define __MDP_VALUE_ITERATION_H__

#include <Eigen/Dense>   // Eigen::Map, Eigen::MatrixXf, Eigen::VectorXf
#include <Eigen/Sparse>  // Eigen::SparseMatrix
#include <fstream>       // std::ofstream
#include <iostream>      // std::ios, std::cout, std::endl
#include <string>        // std::string
#include <tuple>         // std::tie, std::tuple
#include <utility>       // std::pair
#include <vector>        // std::vector

namespace backend {

auto evaluate_mdp(int *pi_data, float *J_data, int *P_indptr, int *P_indices,
                  float *P_data, int P_nnz, int num_nodes, int max_actions,
                  int num_charges, float alpha, float error_min, int num_blocks)
    -> void;

auto async_value_iteration(
    Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> P,
    Eigen::Ref<Eigen::VectorXi> pi, Eigen::Ref<Eigen::VectorXf> J,
    int num_states, int num_nodes, int max_actions, float alpha,
    float error_min, int num_blocks) -> void;

auto update_block(int state_low, int state_high,
                  Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> P,
                  Eigen::Ref<Eigen::VectorXf> J, Eigen::Ref<Eigen::VectorXi> pi,
                  int max_actions, float alpha, int num_nodes, int num_nodes_sq)
    -> float;

auto decode_state(int state, int num_nodes, int num_nodes_sq)
    -> std::tuple<int, int, int>;

auto stage_cost(int charge, int tar_node, int cur_node, int action) -> float;

}  // namespace backend

#endif  // __MDP_VALUE_ITERATION_H__
