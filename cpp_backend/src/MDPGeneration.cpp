#include "MDPGeneration.h"

#include <algorithm>  // std::max, std::min
#include <array>      // std::array
#include <cmath>      // std::floor
#include <iostream>   // std::cout, std::endl
#include <stdexcept>  // std::runtime_error

#include "npy.hpp"

namespace cpp_backend {

auto GenerateMdp(bool *is_charger_data, int *T_indptr, int *T_indices,
                 float *T_data, const int T_nnz, const int num_nodes,
                 const float min_dist, const float max_dist,
                 const int max_actions, char data_out[], const int num_charges,
                 const int max_charge_cost, const bool direct_charge,
                 const float p_travel) -> void {
    const auto num_states = num_nodes * num_nodes * num_charges;

#ifdef VERBOSE
    std::cout << "[cpp backend] (INFO) running" << std::endl;
    std::cout << "[cpp backend] (INFO) num_nodes: " << num_nodes << std::endl;
    std::cout << "[cpp backend] (INFO) num_states: " << num_states << std::endl;
    std::cout << "[cpp backend] (INFO) max_actions: " << max_actions
              << std::endl;
    std::cout << "[cpp backend] (INFO) num_charges: " << num_charges
              << std::endl;
    std::cout << "[cpp backend] (INFO) min_dist: " << min_dist << std::endl;
    std::cout << "[cpp backend] (INFO) max_dist: " << max_dist << std::endl;
#endif

    // Construct transition matrix from existing memory
    Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> T(
        num_nodes, num_nodes, T_nnz, T_indptr, T_indices, T_data);

    // Create probability matrix (state + action -> state)
    const auto [p_row, p_col, p_data] = ConstructP(
        is_charger_data, T, num_states, num_nodes, min_dist, max_dist,
        max_actions, num_charges, max_charge_cost, direct_charge, p_travel);

    // Save probability matrix to data_out in numpy format
#ifdef VERBOSE
    std::cout << "[cpp backend] (INFO) save data" << std::endl;
#endif

    std::array<long unsigned, 1> shape = {p_row.size()};

    try {
        npy::SaveArrayAsNumpy(std::string(data_out) + "/p_row.npy", false,
                              shape.size(), shape.data(), p_row);
    } catch (const std::runtime_error &e) {
        std::cerr
            << "[cpp_backend] (ERROR) Error in SaveArrayAsNumpy for p_row: "
            << e.what() << std::endl;
    }

    shape = {p_col.size()};
    try {
        npy::SaveArrayAsNumpy(std::string(data_out) + "/p_col.npy", false,
                              shape.size(), shape.data(), p_col);
    } catch (const std::runtime_error &e) {
        std::cerr
            << "[cpp_backend] (ERROR) Error in SaveArrayAsNumpy for p_col: "
            << e.what() << std::endl;
    }

    shape = {p_data.size()};
    try {
        npy::SaveArrayAsNumpy(std::string(data_out) + "/p_data.npy", false,
                              shape.size(), shape.data(), p_data);
    } catch (const std::runtime_error &e) {
        std::cerr
            << "[cpp_backend] (ERROR) Error in SaveArrayAsNumpy for p_data: "
            << e.what() << std::endl;
    }
}

auto ConstructP(
    bool *is_charger_data,
    const Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> &T,
    const int num_states, const int num_nodes, const float min_dist,
    const float max_dist, const int max_actions, const int num_charges,
    const int max_charge_cost, const bool direct_charge, const float p_travel)
    -> std::tuple<std::vector<unsigned int>, std::vector<unsigned int>,
                  std::vector<float>> {
    std::vector<unsigned int> p_row;
    std::vector<unsigned int> p_col;
    std::vector<float> p_data;

    // Consider all nodes to move from
    for (auto cur_node = 0; cur_node < num_nodes; ++cur_node) {
        // Gather all next nodes and their required charges
        const auto [next_nodes, charge_costs] =
            get_neighbors(T, is_charger_data, cur_node, min_dist, max_dist,
                          max_charge_cost, direct_charge);

        const auto is_charger = is_charger_data[cur_node];

        // Take action from current node to next node
        for (auto action = 0; action < static_cast<int>(next_nodes.size() + 1);
             ++action) {
            if (action == 0) {
                // Allow to stay at current node: p = 1.0
                UpdateStayNode(&p_row, &p_col, &p_data, cur_node, is_charger,
                               num_nodes, num_charges, max_actions);
            } else {
                // Move to next node + spend charge: 0.0 <= p <= 1.0
                UpdateMoveNode(&p_row, &p_col, &p_data, cur_node, action,
                               num_nodes, num_charges, p_travel, max_actions,
                               next_nodes, charge_costs);
            }
        }
    }

    return {p_row, p_col, p_data};
}

auto UpdateStayNode(std::vector<unsigned int> *p_row,
                    std::vector<unsigned int> *p_col,
                    std::vector<float> *p_values, const int cur_node,
                    const bool is_charger, const int num_nodes,
                    const int num_charges, const int max_actions) -> void {
    // Consider all current charges
    for (auto cur_charge = 0; cur_charge < num_charges; ++cur_charge) {
        // Consider all target nodes
        for (auto tar_node = 0; tar_node < num_nodes; ++tar_node) {
            // Encoded current state
            const auto state =
                get_state(cur_charge, tar_node, cur_node, num_nodes);

            // Add value to probability matrix
            p_row->push_back(state * max_actions);
            p_values->push_back(1.0f);

            if (is_charger) {
                // If current node is charger: Increase charge while staying
                const auto next_state =
                    get_state(std::min(cur_charge + 1, num_charges - 1),
                              tar_node, cur_node, num_nodes);

                p_col->push_back(next_state);
            } else {
                // If current node is non-charger: State is not changing
                p_col->push_back(state);
            }
        }
    }
}

auto UpdateMoveNode(std::vector<unsigned int> *p_row,
                    std::vector<unsigned int> *p_col,
                    std::vector<float> *p_data, const int cur_node,
                    const int action, const int num_nodes,
                    const int num_charges, const float p_travel,
                    const int max_actions, const std::vector<int> &next_nodes,
                    const Eigen::ArrayXi &charge_costs) -> void {
    // Each action will result in one targeted next node
    const auto next_node = next_nodes.at(action - 1);

    // Consider all current charges
    for (auto cur_charge = 0; cur_charge < num_charges; ++cur_charge) {
        // Compute next_charges: Charge >= 0 is considered valid
        const Eigen::ArrayXi next_charges = cur_charge - charge_costs;
        const auto valid = next_charges >= 0;
        const auto num_valid = static_cast<int>(valid.count());

        // Continue if at least one neighbor can be travelled to
        if (num_valid > 0) {
            // Consider all target nodes
            for (auto tar_node = 0; tar_node < num_nodes; ++tar_node) {
                // Encoded current state
                const auto state =
                    get_state(cur_charge, tar_node, cur_node, num_nodes);

                // Next state: Consider all valid next nodes
                for (auto i = 0; i < static_cast<int>(next_nodes.size()); ++i) {
                    if (valid[i]) {
                        // Move probability to considered node: next_nodes.at(i)
                        const auto p = get_move_p(num_valid, next_node,
                                                  next_nodes.at(i), p_travel);

                        // State to move to
                        const auto next_state =
                            get_state(next_charges[i], tar_node,
                                      next_nodes.at(i), num_nodes);

                        // Add value to probability matrix
                        p_row->push_back(state * max_actions + action);
                        p_col->push_back(next_state);
                        p_data->push_back(p);
                    }
                }
            }
        }
    }
}

auto get_neighbors(
    const Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>> &T,
    bool *is_charger_data, const int cur_node, const float min_dist,
    const float max_dist, const int max_charge_cost, const bool direct_charge)
    -> std::pair<std::vector<int>, Eigen::ArrayXi> {
    std::vector<int> nodes;
    std::vector<int> charge_costs;

    // Pre-computation for linear sampling
    float m = 1.0;
    float t = 0.0;

    if (max_dist != min_dist) {
        m = (max_charge_cost - 1) / (max_dist - min_dist);
        t = max_charge_cost - m * max_dist;
    }

    // Loop over non-zero elements in row: cur_node
    for (Eigen::Ref<Eigen::SparseMatrix<float, Eigen::RowMajor>>::InnerIterator
             it(T, cur_node);
         it; ++it) {
        // Add neighbor to output array
        nodes.push_back(static_cast<int>(it.col()));

        // Basic charge cost for neighbor
        auto charge_cost = static_cast<int>(std::floor(m * it.value() + t));

        // Reduced charge cost when moving to charger node
        if (direct_charge && is_charger_data[it.col()]) {
            charge_cost = std::max(charge_cost - 1, 0);
        }

        // Add charge cost to buffer array
        charge_costs.push_back(charge_cost);
    }

    // Create Eigen output array from existing memory
    Eigen::Map<Eigen::ArrayXi> charge_costs_arr(charge_costs.data(),
                                                charge_costs.size());

    return {nodes, charge_costs_arr};
}

auto get_state(const int charge, const int tar_node, const int cur_node,
               const int num_nodes) -> int {
    return num_nodes * num_nodes * charge + num_nodes * tar_node + cur_node;
}

auto get_move_p(const int num_valid, const int next_node,
                const int consider_node, const float p_travel) -> float {
    if (num_valid == 1) {
        return 1.0;
    }

    if (next_node == consider_node) {
        return p_travel;
    }

    return (1.0f - p_travel) / (static_cast<float>(num_valid) - 1.0f);
}

}  // namespace cpp_backend
