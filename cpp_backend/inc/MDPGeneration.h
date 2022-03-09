#ifndef __MDP_GENERATION_H__
#define __MDP_GENERATION_H__

#include <Eigen/Dense>  // Eigen::Map, Eigen::MatrixXf, Eigen::VectorXf, Eigen::ArrayXi
#include <Eigen/Sparse>  // Eigen::SparseMatrix
#include <iostream>      // std::cout, std::endl
#include <tuple>         // std::tuple
#include <utility>       // std::pair
#include <vector>        // std::vector

namespace cpp_backend {

/** \addtogroup MDPGeneration
 *  @{
 */

/**
 * @brief Generate Markov Decision Process by storing a probability matrix in
 * CSR numpy format to data_out directory.
 *
 * @param[in,out] is_charger_data Pointer to list of nodes being chargers
 * @param[in,out] T_indptr Pointer to indptr list of T matrix (np csr)
 * @param[in,out] T_indices Pointer to indices list of T matrix (np csr)
 * @param[in,out] T_data Pointer to data list of T matrix (np csr)
 * @param[in] T_nnz Number of non-zero elements in T matrix (np csr)
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] min_dist Minimum distance between two nodes
 * @param[in] max_dist Maximum distance between two nodes
 * @param[in] max_actions Maximum number of actions in mdp
 * @param[in] data_out Data out directory to store P matrix in
 * @param[in] num_charges Maximum possible charge
 * @param[in] max_charge_cost Charge cost of maximum edge distance max_dist
 * @param[in] direct_charge Directly increase charge when moving to charger
 * @param[in] p_travel Probability of travelling to correct neighbor
 */
auto GenerateMdp(bool *is_charger_data, int *T_indptr, int *T_indices,
                 float *T_data, int T_nnz, int num_nodes, float min_dist,
                 float max_dist, int max_actions, char data_out[],
                 int num_charges, int max_charge_cost, bool direct_charge,
                 float p_travel) -> void;

/**
 * @brief Generate Markov Decision Process and returning it as P_row, P_col, and
 * P_data arrays that can be further processed.
 *
 * @param[in,out] is_charger_data Pointer to list of nodes being chargers
 * @param[in,out] T Reference to constructed T matrix (Eigen csr)
 * @param[in] num_states Total number of states in mdp
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] min_dist Minimum distance between two nodes
 * @param[in] max_dist Maximum distance between two nodes
 * @param[in] max_actions Maximum number of actions in mdp
 * @param[in] num_charges Maximum possible charge
 * @param[in] max_charge_cost Charge cost of maximum edge distance max_dist
 * @param[in] direct_charge Directly increase charge when moving to charger
 * @param[in] p_travel Probability of travelling to correct neighbor
 * @return std::tuple<std::vector<unsigned int>, std::vector<unsigned int>,
 * std::vector<float>>
 */
auto ConstructP(
    bool *is_charger_data,
    const Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> &T,
    int num_states, int num_nodes, float min_dist, float max_dist,
    int max_actions, int num_charges, int max_charge_cost, bool direct_charge,
    float p_travel)
    -> std::tuple<std::vector<unsigned int>, std::vector<unsigned int>,
                  std::vector<float>>;

/**
 * @brief Add values to probability matrix if action dictates to stay at current
 * node. A charge is added if the current node is a charger.
 *
 * @param[in,out] p_row Pointer to row array in P matrix (np csr)
 * @param[in,out] p_col Pointer to column array in P matrix (np csr)
 * @param[in,out] p_data Pointer to data array in P matrix (np csr)
 * @param[in] cur_node Current node of state
 * @param[in] is_charger Current node is a charger
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] num_charges Maximum possible charge
 * @param[in] max_actions Maximum number of actions in mdp
 */
auto UpdateStayNode(std::vector<unsigned int> *p_row,
                    std::vector<unsigned int> *p_col,
                    std::vector<float> *p_data, int cur_node, bool is_charger,
                    int num_nodes, int num_charges, int max_actions) -> void;

/**
 * @brief Add values to probability matrix if action dictates to move to
 * neighboring nodes. Charge is lost based on charge costs of neighboring nodes.
 *
 * @param[in,out] p_row Pointer to row array in P matrix (np csr)
 * @param[in,out] p_col Pointer to column array in P matrix (np csr)
 * @param[in,out] p_data Pointer to data array in P matrix (np csr)
 * @param[in] cur_node Current node of state
 * @param[in] action Current action
 * @param[in] num_nodes Total number of nodes in mdp
 * @param[in] num_charges Maximum possible charge
 * @param[in] p_travel Probability of travelling to correct neighbor
 * @param[in] max_actions Maximum number of actions in mdp
 * @param[in] next_nodes List of possible next nodes
 * @param[in,out] charge_costs Reference to charge costs array
 */
auto UpdateMoveNode(std::vector<unsigned int> *p_row,
                    std::vector<unsigned int> *p_col,
                    std::vector<float> *p_data, int cur_node, int action,
                    int num_nodes, int num_charges, float p_travel,
                    int max_actions, const std::vector<int> &next_nodes,
                    const Eigen::ArrayXi &charge_costs) -> void;

/**
 * @brief Get neighbor nodes and required charges from current node in state.
 *
 * @param[in] T Reference to constructed Eigen transition matrix (csr)
 * @param[in,out] is_charger_data Pointer to list of nodes being chargers
 * @param[in] cur_node Current node of state
 * @param[in] min_dist Minimum distance between two nodes
 * @param[in] max_dist Maximum distance between two nodes
 * @param[in] max_charge_cost Charge cost of maximum edge distance max_dist
 * @param[in] direct_charge Directly increase charge when moving to charger
 * @return std::pair<std::vector<int>, Eigen::ArrayXi> (neighbor nodes, charge
 * costs)
 */
auto get_neighbors(
    const Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> &T,
    bool *is_charger_data, int cur_node, float min_dist, float max_dist,
    int max_charge_cost, bool direct_charge)
    -> std::pair<std::vector<int>, Eigen::ArrayXi>;

/**
 * @brief Encode state from tuple (cur_charge, tar_node, cur_node) into number.
 *
 * @param[in] cur_charge Current charge in state
 * @param[in] tar_node Target node in state
 * @param[in] cur_node Current node in state
 * @param[in] num_nodes Total number of nodes in mdp
 * @return int State number
 */
auto get_state(int charge, int tar_node, int cur_node, int num_nodes) -> int;

/**
 * @brief Get move probability to consider_node given that next_node is target.
 *
 * @param[in] num_valid Total number of valid next nodes that can be travelled
 * to
 * @param[in] next_node Target next node to travel to
 * @param[in] consider_node Considered node to travel to (Not necessarily equal
 * to next_node)
 * @param[in] p_travel Probability of travelling to correct neighbor
 * @return float Probability to travel to consider_node
 */
auto get_move_p(int num_valid, int next_node, int consider_node, float p_travel)
    -> float;

/** @}*/

}  // namespace cpp_backend

#endif  // __MDP_VALUE_ITERATION_H__
