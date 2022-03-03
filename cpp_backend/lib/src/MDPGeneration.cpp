#include "MDPGeneration.h"

#include <algorithm>  // sdt::max_element
#include <array>      // std::array
#include <cmath>      // std::floor, std::max, std::min, std::exp, std::abs
#include <iostream>   // std::cout, std::endl
#include <numeric>    // std::iota

#include "npy.hpp"
#include "omp.h"

namespace backend {

auto generate_mdp(bool *is_charger_data, int *T_indptr, int *T_indices,
                  float *T_data, const int T_nnz, const int num_nodes,
                  const float min_dist, const float max_dist,
                  const int max_actions, char data_out[], const int num_charges,
                  const int max_charge, const float sigma_env,
                  const float p_travel, const int n_charge) -> void {
#ifdef VERBOSE
    std::cout << "[cpp backend] running" << std::endl;
#endif

    const auto num_states = num_nodes * num_nodes * num_charges;

    std::cout << "[cpp backend] num_nodes: " << num_nodes << std::endl;
    std::cout << "[cpp backend] num_states: " << num_states << std::endl;
    std::cout << "[cpp backend] max_actions: " << max_actions << std::endl;
    std::cout << "[cpp backend] num_charges: " << num_charges << std::endl;

    // Construct T matrix from existing memory
    Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> T(
        num_nodes, num_nodes, T_nnz, T_indptr, T_indices, T_data);

    // Construct probability matrix (state + action, state)
    std::vector<unsigned int> p_row;
    std::vector<unsigned int> p_col;
    std::vector<float> p_data;

    std::tie(p_row, p_col, p_data) = construct_p(
        is_charger_data, T, num_states, num_nodes, min_dist, max_dist,
        max_actions, num_charges, max_charge, sigma_env, p_travel, n_charge);

// Save probability matrix data
#ifdef VERBOSE
    std::cout << "[cpp backend] save data" << std::endl;
#endif

    std::array<long unsigned, 1> shape = {p_row.size()};
    npy::SaveArrayAsNumpy(std::string(data_out) + "/p_row.npy", false,
                          shape.size(), shape.data(), p_row);

    shape = {p_col.size()};
    npy::SaveArrayAsNumpy(std::string(data_out) + "/p_col.npy", false,
                          shape.size(), shape.data(), p_col);

    shape = {p_data.size()};
    npy::SaveArrayAsNumpy(std::string(data_out) + "/p_data.npy", false,
                          shape.size(), shape.data(), p_data);
}

auto construct_p(bool *is_charger_data,
                 Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> T,
                 const int num_states, const int num_nodes,
                 const float min_dist, const float max_dist,
                 const int max_actions, const int num_charges,
                 const int max_charge, const float sigma_env,
                 const float p_travel, const int n_charge)
    -> std::tuple<std::vector<unsigned int>, std::vector<unsigned int>,
                  std::vector<float>> {
    std::vector<unsigned int> p_row;
    std::vector<unsigned int> p_col;
    std::vector<float> p_data;

    // Consider all nodes to move from
    for (int cur_node = 0; cur_node < num_nodes; ++cur_node) {
        // Current node is charger
        const auto is_charger = is_charger_data[cur_node];

        // Next nodes to travel to, Distance to next nodes and associated charge
        // consumption per distance
        std::vector<int> next_nodes;
        std::vector<float> distances;

        std::tie(next_nodes, distances) = get_neighbors(T, cur_node);
        const auto actions = static_cast<int>(next_nodes.size()) + 1;
        const auto charges =
            get_charges(distances, min_dist, max_dist, max_charge);

        for (int action = 0; action < max_actions; ++action) {
            // Stay at current node, Gain +1 charge on charger nodes
            if (action == 0) {
                p_stay_node(&p_row, &p_col, &p_data, cur_node, is_charger,
                            num_nodes, num_charges, max_actions);
            }

            // Move to next node & Spend charge based on distance to next_node,
            // Consider all neighbor nodes: Compute probability of travelling to
            // next_node using certain charge
            if (action > 0 && action < actions) {
                p_move_node(&p_row, &p_col, &p_data, cur_node, action,
                            num_nodes, num_charges, p_travel, sigma_env,
                            n_charge, max_actions, next_nodes, charges);
            }
        }
    }

    return {p_row, p_col, p_data};
}

auto p_stay_node(std::vector<unsigned int> *p_row,
                 std::vector<unsigned int> *p_col, std::vector<float> *p_values,
                 const int cur_node, const bool is_charger, const int num_nodes,
                 int num_charges, const int max_actions) -> void {
    // Consider all possible charges
    for (int charge = 0; charge < num_charges; ++charge) {
        // Consider all target nodes
        for (int tar_node = 0; tar_node < num_nodes; ++tar_node) {
            // State to move from
            const auto state =
                encode_state(charge, tar_node, cur_node, num_nodes);

            p_row->push_back(state * max_actions);
            p_values->push_back(1.0f);

            if (is_charger) {
                // State to move to: +1 charge if currently on charger
                const auto new_state =
                    encode_state(std::min(charge + 1, num_charges - 1),
                                 tar_node, cur_node, num_nodes);

                p_col->push_back(new_state);
            } else {
                p_col->push_back(state);
            }
        }
    }
}

auto p_move_node(std::vector<unsigned int> *p_row,
                 std::vector<unsigned int> *p_col, std::vector<float> *p_data,
                 const int cur_node, const int action, const int num_nodes,
                 const int num_charges, const float p_travel,
                 const float sigma_env, const int n_charge,
                 const int max_actions, const std::vector<int> &next_nodes,
                 const std::vector<int> &charges) -> void {
    // Action: Travel to next_node spending next_charge
    const auto next_node = next_nodes.at(action - 1);
    const auto next_charge = charges.at(action - 1);

    // Next_nodes distribution: Consider all neighbors with certain probability
    // Uniform distribution: next node with probability p_travel
    const auto p_next_nodes = get_uni_dist(next_nodes, next_node, p_travel);

    // Next_charges: Consider n_charge neighbors to both sides to next_charge
    // with certain probability
    // Normal distribution: next_charge with highest probability
    // (Environment uncertainties: sigma_env)
    std::vector<int> next_charges(1 + n_charge +
                                  std::min(next_charge - 1, n_charge));
    std::iota(next_charges.begin(), next_charges.end(),
              std::max(1, next_charge - n_charge));

    const auto p_next_charges =
        get_norm_dist(next_charges, next_charge, sigma_env);

    // Combine distributions: Probability (next_charge, next_node)
    Eigen::MatrixXf p_mat = p_next_charges * p_next_nodes.transpose();
    p_mat /= p_mat.sum();

    // Consider all possible charges
    for (int cur_charge = 0; cur_charge < num_charges; ++cur_charge) {
        std::vector<int> total_next_charges;

        // Update next_charges based on current charge
        for (size_t i = 0; i < next_charges.size(); ++i) {
            total_next_charges.push_back(
                std::max(0, cur_charge - next_charges.at(i)));
        }

        // Consider all possible target nodes
        for (int tar_node = 0; tar_node < num_nodes; ++tar_node) {
            // State to move from
            const auto state =
                encode_state(cur_charge, tar_node, cur_node, num_nodes);

            // Next state: Consider all (next_charge, next_node) in computed
            // probability distribution
            for (size_t i = 0; i < total_next_charges.size(); ++i) {
                for (size_t j = 0; j < next_nodes.size(); ++j) {
                    // State to move to
                    const auto next_state =
                        encode_state(total_next_charges.at(i), tar_node,
                                     next_nodes.at(j), num_nodes);

                    // Append data: state & action -> p_mat(i_charge, j_node)
                    p_row->push_back(state * max_actions + action);
                    p_col->push_back(next_state);
                    p_data->push_back(p_mat(i, j));
                }
            }
        }
    }
}

auto get_neighbors(Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> T,
                   const int node)
    -> std::pair<std::vector<int>, std::vector<float>> {
    std::vector<int> nodes;
    std::vector<float> distances;

    for (Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>>::InnerIterator
             it(T, node);
         it; ++it) {
        nodes.push_back(it.col());
        distances.push_back(it.value());
    }

    return {nodes, distances};
}

auto encode_state(const int charge, const int tar_node, const int cur_node,
                  const int num_nodes) -> int {
    return num_nodes * num_nodes * charge + num_nodes * tar_node + cur_node;
}

auto get_charges(const std::vector<float> &distances, const float min_dist,
                 const float max_dist, const int max_charge)
    -> std::vector<int> {
    std::vector<int> charges(distances.size());

    // Linear sampling
    const auto m = (max_charge - 1) / (max_dist - min_dist);
    const auto t = max_charge - m * max_dist;

    for (size_t i = 0; i < distances.size(); ++i) {
        charges.at(i) = static_cast<int>(std::floor(m * distances.at(i) + t));
    }

    return charges;
}

auto get_norm_dist(const std::vector<int> &values, const int exp,
                   const float sigma) -> Eigen::VectorXf {
    Eigen::VectorXf norm_dist(values.size());

    const auto d = 2.0f * sigma * sigma;
    auto p_total = 0.0f;

    // Fill vector with normal distribution
    for (size_t i = 0; i < values.size(); ++i) {
        const auto diff = static_cast<float>(values.at(i) - exp);
        const auto p = std::exp(-diff * diff / d);

        norm_dist(i) = p;
        p_total += p;
    }

    // Multiply each member with normalization factor
    norm_dist *= static_cast<float>(1.0f / p_total);

    return norm_dist;
}

auto get_uni_dist(const std::vector<int> &values, const int exp, const float p)
    -> Eigen::VectorXf {
    Eigen::VectorXf uni_dist(values.size());

    // Expected member with probability p
    // Leftover probability (1 - p) is shared between all other members
    const auto p_ = (1.0 - p) / static_cast<float>(values.size() - 1);

    for (size_t i = 0; i < values.size(); ++i) {
        if (values.at(i) == exp) {
            uni_dist(i) = p;
        } else {
            uni_dist(i) = p_;
        }
    }

    return uni_dist;
}

}  // namespace backend
