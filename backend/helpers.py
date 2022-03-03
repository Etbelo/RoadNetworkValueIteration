import logging

import numpy as np
from scipy.sparse import csr_matrix


def travel(state, action, P, max_actions):

    row_in_p = state * max_actions + action

    next_states = P[row_in_p].nonzero()[1]

    if next_states.size == 0:
        raise ValueError('State has no successors')

    p_next_states = [P[row_in_p, next_state] for next_state in next_states]

    p_total = np.sum(p_next_states)

    if 1.0 < p_total < 1.1:
        p_next_states = [p / p_total for p in p_next_states]

    return np.random.choice(next_states, p=p_next_states)


def path_from_policy(state, P, pi, num_nodes, max_actions):

    iter = 0

    charge, tar_node, cur_node = decode_state(state, num_nodes)
    path_states = [state]
    path_nodes = [cur_node]

    goal_reached = False

    while not goal_reached and iter < 200 and charge != 0:
        iter += 1

        try:
            state = travel(state, pi[state], P, max_actions)
            charge, tar_node, cur_node = decode_state(state, num_nodes)
            goal_reached = cur_node == tar_node
            path_states.append(state)
            path_nodes.append(cur_node)
        except ValueError as e:
            logging.error(f'No result in path_from_policy due to error: {e}')
            return path_states, path_nodes

    return path_states, path_nodes


def encode_state(charge, tar_node, cur_node, num_nodes):

    return charge * (num_nodes ** 2) + tar_node * num_nodes + cur_node


def decode_state(state, num_nodes):

    charge = state // (num_nodes ** 2)
    tar_node = state % (num_nodes ** 2) // num_nodes
    cur_node = state % (num_nodes ** 2) % num_nodes

    return charge, tar_node, cur_node


def get_random_chargers(num_nodes, num_chargers):

    charger_ids = np.random.choice(a=np.arange(
        num_nodes), size=min(num_nodes // 2, num_chargers), replace=False)

    chargers = np.zeros(num_nodes, dtype=bool)
    chargers[charger_ids] = True

    return chargers, charger_ids


def get_max_u(T):

    num_transitions = (T > 0.0).sum(0)

    max_u = np.max(num_transitions)
    max_node = np.argmax(num_transitions)

    return int(max_u + 1), int(max_node)


def get_transition_matrix(num_nodes, start_ids, end_ids, distances):

    # Make matrix bi-directional
    data = np.hstack((distances, distances))
    row = np.hstack((start_ids, end_ids))
    col = np.hstack((end_ids, start_ids))

    # Transition distance between nodes
    T = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    return T


def compress_csr_graph(T):

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

    return T_new, row, col, valid, vertices.size
