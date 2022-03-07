#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <ios>
#include <iostream>
#include <tuple>
#include <utility>

#include "MDPValueIteration.h"
#include "npy.hpp"

int main() {
    std::vector<int> charges = {2, 3, 2};
    Eigen::Map<Eigen::ArrayXi> charge_costs(charges.data(), charges.size());

    std::cout << "charge costs" << std::endl;
    for (const auto& it : charge_costs) {
        std::cout << it << std::endl;
    }

    Eigen::ArrayXi next_charges = 2 - charge_costs;

    const auto valid = next_charges >= 0;
    const auto num_valid = static_cast<int>(valid.count());

    std::cout << "valid" << std::endl;

    for (const auto& it : valid) {
        std::cout << it << std::endl;
    }

    std::cout << "num_valid: " << num_valid << std::endl;

    return 0;
}