#ifndef __MDP_GENERATION_H__
#define __MDP_GENERATION_H__

#include <Eigen/Dense>   // Eigen::Map, Eigen::MatrixXf, Eigen::VectorXf
#include <Eigen/Sparse>  // Eigen::SparseMatrix
#include <iostream>      // std::cout, std::endl
#include <tuple>         // std::tie, std::tuple
#include <utility>       // std::pair
#include <vector>        // std::vector

namespace backend {

auto generate_mdp(bool *is_charger_data, int *T_indptr, int *T_indices,
                  float *T_data, const int T_nnz, const int num_nodes,
                  const float min_dist, const float max_dist,
                  const int max_actions, char data_out[], const int num_charges,
                  const int max_charge, const float sigma_env,
                  const float p_travel, const int n_charge) -> void;

auto construct_p(bool *is_charger_data,
                 Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> T,
                 int num_states, int num_nodes, float min_dist, float max_dist,
                 int max_actions, int num_charges, int max_charge,
                 float sigma_env, float p_travel, int n_charge)
    -> std::tuple<std::vector<unsigned int>, std::vector<unsigned int>,
                  std::vector<float>>;

auto p_stay_node(std::vector<unsigned int> *p_row,
                 std::vector<unsigned int> *p_col, std::vector<float> *p_data,
                 int cur_node, bool is_charger, int num_nodes, int num_charges,
                 int max_actions) -> void;

auto p_move_node(std::vector<unsigned int> *p_row,
                 std::vector<unsigned int> *p_col, std::vector<float> *p_data,
                 int cur_node, int action, int num_nodes, int num_charges,
                 float p_travel, float sigma_env, int n_charge, int max_actions,
                 const std::vector<int> &next_nodes,
                 const std::vector<int> &charges) -> void;

auto get_neighbors(Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> T,
                   int node) -> std::pair<std::vector<int>, std::vector<float>>;

auto encode_state(int charge, int tar_node, int cur_node, int num_nodes) -> int;

auto get_charges(const std::vector<float> &distances, float min_dist,
                 float max_dist, int max_charge) -> std::vector<int>;

auto get_norm_dist(const std::vector<int> &values, int exp, float sigma)
    -> Eigen::VectorXf;

auto get_uni_dist(const std::vector<int> &values, int exp, float p)
    -> Eigen::VectorXf;
}  // namespace backend

#endif  // __MDP_VALUE_ITERATION_H__
