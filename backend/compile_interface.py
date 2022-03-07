from cffi import FFI


def compile_interface(verbose=True):
    '''! Define interface to c++ backend functions as defined in Interface.h

    @param verbose Print compile output to the console
    '''

    ffi = FFI()

    ffi.cdef("""void cffi_generate_mdp(bool *is_charger_data, int *T_indptr,
                              int *T_indices, float *T_data, int T_nnz,
                              int num_nodes, float min_dist, float max_dist,
                              int max_actions, char data_out[], int num_charges,
                              int max_charge, float p_travel);""")

    ffi.cdef("""void cffi_evaluate_mdp(int *pi_data, float *J_data, int *P_indptr,
                              int *P_indices, float *P_data, int P_nnz,
                              int num_nodes, int max_actions, int num_charges,
                              float alpha, float error_min, int num_blocks);""")

    ffi.set_source(
        "cpp_interface", """ #include "Interface.h" """, include_dirs=['inc'],
        libraries=['backend'],
        library_dirs=['lib'],
        extra_link_args=['-Wl,-rpath=$ORIGIN/lib', '-fopenmp'],
        extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp', '-D use_openmp'],
        source_extension='.cpp')

    return ffi.compile(verbose=verbose)


if __name__ == "__main__":
    compile_interface()
