#include "MDPGeneration.h"
#include "MDPValueIteration.h"

extern "C" {
extern void cffi_generate_mdp(bool *is_charger_data, int *T_indptr,
                              int *T_indices, float *T_data, int T_nnz,
                              int num_nodes, float min_dist, float max_dist,
                              int max_actions, char data_out[], int num_charges,
                              int max_charge, float sigma_env, float p_travel,
                              int n_charge) {
    backend::generate_mdp(is_charger_data, T_indptr, T_indices, T_data, T_nnz,
                          num_nodes, min_dist, max_dist, max_actions, data_out,
                          num_charges, max_charge, sigma_env, p_travel,
                          n_charge);
}

extern void cffi_evaluate_mdp(int *pi_data, float *J_data, int *P_indptr,
                              int *P_indices, float *P_data, int P_nnz,
                              int num_nodes, int max_actions, int num_charges,
                              float alpha, float error_min, int num_blocks) {
    backend::evaluate_mdp(pi_data, J_data, P_indptr, P_indices, P_data, P_nnz,
                          num_nodes, max_actions, num_charges, alpha, error_min,
                          num_blocks);
}
}