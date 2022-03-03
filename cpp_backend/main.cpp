#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <iostream>
#include <tuple>
#include <utility>

#include "MDPValueIteration.h"
#include "npy.hpp"

// Testing
auto test(std::vector<Eigen::Triplet<float>> *p_data) -> void {
    p_data->push_back({0, 0, 0.5});
    p_data->push_back({0, 1, 0.5});
    p_data->push_back({1, 2, 1.0});
    p_data->push_back({2, 1, 0.5});
    p_data->push_back({2, 2, 0.5});
}

auto flatten_triplet(std::vector<Eigen::Triplet<float>> *p_data)
    -> std::pair<std::vector<float>, std::array<long unsigned, 2>> {
    const auto size = p_data->size();

    std::vector<float> data;
    std::array<long unsigned, 2> shape{{size, 3}};

    for (size_t i = 0; i < size; ++i) {
        const auto first = p_data->at(0);
        data.push_back(first.row());
        data.push_back(first.col());
        data.push_back(first.value());
        p_data->erase(p_data->begin() + 0);
        p_data->shrink_to_fit();
    }

    p_data->clear();
    p_data->shrink_to_fit();

    return {data, shape};
}

int main() {
    // std::vector<Eigen::Triplet<float>> p_data;

    // test(&p_data);

    // std::vector<float> data;
    // std::array<long unsigned, 2> shape;

    // std::tie(data, shape) = flatten_triplet(&p_data);

    // npy::SaveArrayAsNumpy("data_out/out.npy", false, shape.size(),
    // shape.data(),
    //                       data);

    std::vector<int> indptr = {0, 3, 3, 6};
    std::vector<int> indices = {0, 1, 2, 0, 1, 2};
    std::vector<float> values = {0.33, 0.33, 0.33, 0.33, 0.33, 0.33};

    // Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> mat(
    //     3, 3, values.size(), indptr.data(), indices.data(), values.data());

    std::array<long unsigned, 1> shape = {indptr.size()};

    npy::SaveArrayAsNumpy("data_out/indptr.npy", false, shape.size(),
                          shape.data(), indptr);

    shape = {indices.size()};

    npy::SaveArrayAsNumpy("data_out/indices.npy", false, shape.size(),
                          shape.data(), indices);

    shape = {values.size()};

    npy::SaveArrayAsNumpy("data_out/values.npy", false, shape.size(),
                          shape.data(), values);

    return 0;
}