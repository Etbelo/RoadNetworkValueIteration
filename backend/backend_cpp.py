import os
import logging

import numpy as np


logger = logging.Logger('python')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(name)s] (%(levelname)s) %(message)s'))

logger.addHandler(ch)


##################
# Interface Code #
##################


def _find_compile_output():
    """ Returns all the files that belong to the compiled interface """

    def should_include(file):
        return file.startswith(
            "cpp_interface") and file.endswith((".cpp", ".o", ".so"))

    return [file for file in os.listdir(os.path.dirname(__file__)) if should_include(file)]


def _compile_output_complete():
    return len(_find_compile_output()) == 3


if not _compile_output_complete():
    logger.info("compile c++ backend")

    base_dir = os.path.dirname(__file__)

    inc_dir = os.path.join(base_dir, 'inc')
    lib_dir = os.path.join(base_dir, 'lib')

    if not os.path.exists(inc_dir) or not os.path.exists(lib_dir) or len(
            os.listdir(inc_dir)) == 0 or len(
            os.listdir(lib_dir)) == 0:
        raise ImportError(
            "Compiled backend library cannot be found. Please make sure to run make compile!")

    os.chdir(base_dir)

    from .compile_interface import compile_interface

    compile_interface(verbose=True)

    os.chdir("..")

if not _compile_output_complete():
    logger.error("Compilation output is incomplete")
    raise ImportError("Compiling the c++ python interface was not successful")

try:
    from . import cpp_interface
    from cffi import FFI

except (ModuleNotFoundError, ImportError) as e:
    logger.error(f"Compilation error: {e}")
    raise

_ffi = FFI()


###############
# C++ Wrapper #
###############


def generate_mdp(
        chargers, T, num_nodes, min_dist, max_dist, max_actions, data_out, num_charges,
        max_charge_cost, direct_charge, p_travel):
    '''! Call cpp backend function to generate a Markov Decision Process.

    @param chargers List of nodes being chargers
    @param T Transition matrix in scipy csr format
    @param num_nodes Total number of nodes in mdp
    @param min_dist Minimum distance between two nodes
    @param max_dist Maximum distance between two nodes
    @param max_actions Maximum number of possible actions in mdp
    @param data_out Data out directory to store results in
    @param num_chargers Total number of maximum possible charges
    @param max_charge_cost Maximum charge cost associated with max_dist
    @param direct_charge Directly increase charge when moving to charger
    @param p_travel Probability of travelling to correct neighbor
    '''

    # Transition matrix
    T_indptr = _ffi.cast("int*", T.indptr.ctypes.data)
    T_indices = _ffi.cast("int*", T.indices.ctypes.data)
    T_data = _ffi.cast("float*", T.data.ctypes.data)

    # List of charger nodes
    is_charger_data = _ffi.cast("bool*", chargers.ctypes.data)

    # Directory for data dumps
    data_out_data = _ffi.new("char[]", data_out.encode())

    cpp_interface.lib.cffi_generate_mdp(
        is_charger_data, T_indptr, T_indices, T_data, T.data.size, num_nodes, min_dist, max_dist,
        max_actions, data_out_data, num_charges, max_charge_cost, direct_charge, p_travel)


def evaluate_mdp(num_states, P, num_nodes, max_actions, num_charges, alpha, error_min, num_blocks):
    '''! Call cpp backend to evaluate Markov Decision Process and return the policy and total cost arrays.

    @param num_states Total number of states in mdp
    @param P Probability matrix in scipy csr format
    @param num_nodes Total number of nodes in mdp
    @param max_actions Maximum number of possible actions in mdp
    @param num_charges Total number of maximum possible charges
    @param alpha Discount factor for value iteration
    @param error_min Minimum error as target for greedy algorithm
    @param num_blocks Number of statespace blocks for threads to work on

    @return (policy array, cost array) in numpy format for all states
    '''

    # Transition matrix
    P_indptr = _ffi.cast("int*", P.indptr.ctypes.data)
    P_indices = _ffi.cast("int*", P.indices.ctypes.data)
    P_data = _ffi.cast("float*", P.data.ctypes.data)

    # Evaluated policy output
    pi = np.zeros(num_states, dtype=np.int32)
    pi_data = _ffi.cast("int*", pi.ctypes.data)

    # Evaluated cost
    J = np.zeros(num_states, dtype=np.float32)
    J_data = _ffi.cast("float*", J.ctypes.data)

    # Call evaluation of mdp
    cpp_interface.lib.cffi_evaluate_mdp(
        pi_data, J_data, P_indptr, P_indices, P_data, P.data.size, num_nodes, max_actions,
        num_charges, alpha, error_min, num_blocks)

    # Return policy and action cost
    return pi, J
