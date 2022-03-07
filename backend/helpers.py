import logging
import os

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.Logger('python')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(name)s] (%(levelname)s) %(message)s'))

logger.addHandler(ch)


def files_available(data_dir, files):
    '''! Test if all files are available in data_dir directory.

    @param data_dir Directory to search for files in
    @param files List of files to search for in data_dir directory

    @return bool valid, list of missing files
    '''

    valid = True
    missing = []

    for file in files:
        path = os.path.join(data_dir, file)
        file_valid = os.path.isfile(path)
        valid &= file_valid
        if not file_valid:
            missing.append(path)

    return valid, missing


def travel(state, action, P, max_actions, static):
    '''! Travel from state to next state given an action.

    @param state State to travel from
    @param action Action to take from state
    @param P Probability matrix for (state + action -> state)
    @param max_actions Maximum number of actions possible in mdp
    @param static True: Static evaluation (Direct policy), False: Stochastic evluation 
                    (Including stochastic next state)

    @param return Next state
    '''

    row_in_p = state * max_actions + action

    next_states = P[row_in_p].nonzero()[1]

    if next_states.size == 0:
        raise ValueError('State has no successors')

    p_next_states = [P[row_in_p, next_state] for next_state in next_states]

    # Static next state
    if static:
        return next_states[np.argmax(p_next_states)]

    # Stochastic next state
    p_total = np.sum(p_next_states)
    p_next_states = [p / p_total for p in p_next_states]

    return np.random.choice(next_states, p=p_next_states)


def path_from_policy(state, P, pi, num_nodes, max_actions, max_iter, static):
    '''! Generate a path from given policy and state.

    @param state State to start from
    @param P Probability matrix for (state + action -> state)
    @param pi Policy (List of actions for each state)
    @param num_nodes Total number of nodes in the mdp
    @param max_actions Maximum number of actions possible in mdp
    @param max_iter Maximum number of allowed iterations until breaking the policy loop
    @param static True: Static evaluation (Direct policy), False: Stochastic evluation 
                    (Including stochastic next state)

    @return List of states, List of nodes
    '''

    iter = 0

    charge, tar_node, cur_node = decode_state(state, num_nodes)
    path_states = [state]
    path_nodes = [cur_node]

    goal_reached = False

    while not goal_reached and iter < max_iter and charge != 0:
        iter += 1

        try:
            state = travel(state, pi[state], P, max_actions, static)
            charge, tar_node, cur_node = decode_state(state, num_nodes)
            goal_reached = cur_node == tar_node
            path_states.append(state)
            path_nodes.append(cur_node)
        except ValueError as e:
            logger.error(f'No result in path_from_policy due to error: {e}')
            return path_states, path_nodes

    return path_states, path_nodes


def encode_state(charge, tar_node, cur_node, num_nodes):
    '''! Encode state tuple to one number.

    @param charge Current charge in state
    @param tar_node Target node to find
    @param cur_node Current node in state
    @param num_nodes Total number of nodes in the mdp

    @return State number
    '''

    return charge * (num_nodes ** 2) + tar_node * num_nodes + cur_node


def decode_state(state, num_nodes):
    '''! Decode state number to tuple.

    @param state State number to decode
    @param num_nodes Total number of nodes in the mdp

    @param Tuple of state elements (charge, tar_node, cur_node)
    '''

    charge = state // (num_nodes ** 2)
    tar_node = state % (num_nodes ** 2) // num_nodes
    cur_node = state % (num_nodes ** 2) % num_nodes

    return charge, tar_node, cur_node


def get_random_chargers(num_nodes, num_chargers):
    '''! Create list of random chargers in the current mdp.

    @param num_nodes Total number of nodes in the mdp
    @param num_chargers Total number of required random chargers

    @return List of node ids that have been declared chargers
    '''

    charger_ids = np.random.choice(a=np.arange(
        num_nodes), size=min(num_nodes // 2, num_chargers), replace=False)

    chargers = np.zeros(num_nodes, dtype=bool)
    chargers[charger_ids] = True

    return chargers, charger_ids


def get_max_u(T):
    '''! Get the maximum number of actions possible in the current mdp.

    @param T Transition matrix (node -> node)

    @return Number of maximum actions (Including action = 0: Staying) 
    '''

    num_transitions = (T > 0.0).sum(0)

    max_u = np.max(num_transitions)
    max_node = np.argmax(num_transitions)

    return int(max_u + 1), int(max_node)


def get_transition_matrix(num_nodes, T_row, T_col, T_dist):
    '''! Create transition matrix from edgelists.

    @param num_nodes Total number of nodes in the mdp
    @param T_row Edgelist from
    @param T_col Edgelist to
    @param T_dist Distance of edges_from -> edges_to

    @return Transition matrix in sparse CSR format
    '''

    # Make matrix bi-directional
    data = np.hstack((T_dist, T_dist))
    row = np.hstack((T_row, T_col))
    col = np.hstack((T_col, T_row))

    # Transition distance between nodes
    return csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)


def compress_csr_graph(T):
    '''! Compress sparse CSR matrix by removing degree 2 nodes.

    @param T Transition matrix (node -> node)

    @return New transition matrix in CSR format, new edgelist_from, new egdelist_to, 
            new number of nodes
    '''

    T_data = np.zeros((0, 3))

    deg = np.asarray((T != 0).sum(0)).flatten()
    valid = np.logical_or(deg == 1, deg > 2)
    new_nodes = np.cumsum(valid) - 1
    vertices = np.where(valid)[0]

    for vertex in vertices:
        neighbors = T[vertex, :].nonzero()[1]
        vertex_data = np.zeros((0, 3))

        for neighbor in neighbors:
            prev_node = vertex
            node = neighbor

            total_dist = T[prev_node, node]

            while deg[node] == 2:
                path_nodes = T[node, :].nonzero()[1]
                new_node = path_nodes[np.where(path_nodes != prev_node)[0][0]]

                prev_node = node
                node = new_node

                total_dist += T[prev_node, node]

            if node != vertex:
                if new_nodes[node] not in vertex_data[:, 1]:
                    vertex_data = np.vstack(
                        (vertex_data, [new_nodes[vertex], new_nodes[node], total_dist]))

                else:
                    node_ind = np.where(
                        vertex_data[:, 1] == new_nodes[node])[0][0]
                    vertex_data[node_ind, 2] = min(
                        vertex_data[node_ind, 2], total_dist)

        T_data = np.vstack((T_data, vertex_data))

    row = T_data[:, 0].astype(dtype=np.int32)
    col = T_data[:, 1].astype(dtype=np.int32)
    data = T_data[:, 2].astype(dtype=np.float32)

    T_new = csr_matrix((data, (row, col)), shape=(vertices.size, vertices.size))

    return T_new, row, col, valid, new_nodes, vertices.size
