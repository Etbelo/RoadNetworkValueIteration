import time
import argparse
import os

import numpy as np
from scipy.sparse import csr_matrix
import pickle
import yaml

from backend.backend_cpp import *
from backend.helpers import *
from backend.visualization import *


logger = logging.Logger('python')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(name)s] (%(levelname)s) %(message)s'))

logger.addHandler(ch)


def main(plot, show_fig, force, config_file, data_out):

    logger.info('running')

    # -----------------------------------------------------------
    # Read Parameters
    # -----------------------------------------------------------

    params = {}

    with open(config_file, 'rb') as file:
        try:
            params = yaml.safe_load(file)
        except Exception as e:
            logger.error(f'Parameter exception: {e}')

    valid, missing = files_available(
        data_out, ['p_params.p', 'p_row.npy', 'p_col.npy', 'p_data.npy'])

    if not valid:
        logger.error(f'Missing files: {missing}')
        return

    data = pickle.load(open(os.path.join(data_out, 'p_params.p'), "rb"))
    logger.info(f'p_data: {data}')

    p_row = np.load(os.path.join(data_out, 'p_row.npy'))
    p_col = np.load(os.path.join(data_out, 'p_col.npy'))
    p_data = np.load(os.path.join(data_out, 'p_data.npy'))

    P = csr_matrix((p_data, (p_row, p_col)), shape=data['p_shape'])

    # -----------------------------------------------------------
    # Evaluate MDP
    # -----------------------------------------------------------

    policy_exists = os.path.isfile(os.path.join(data_out, 'pi.npy'))

    if not policy_exists or force:

        # Call backend
        pi, J = evaluate_mdp(
            data['num_states'],
            P, data['num_nodes'],
            data['max_actions'],
            params['generate']['num_charges'],
            params['evaluate']['alpha'],
            params['evaluate']['error_min'],
            params['evaluate']['num_blocks'])

        np.save(os.path.join(data_out, 'pi'), pi)
        np.save(os.path.join(data_out, 'J'), J)

    if policy_exists and not force:
        pi = np.load(os.path.join(data_out, 'pi.npy'))

    # -----------------------------------------------------------
    # Test Policy
    # -----------------------------------------------------------

    # Test specific state using the aquired policy
    start_charge = params['test']['start_charge']
    start_tar_node = params['test']['target_node']
    start_cur_node = params['test']['start_node']

    state = encode_state(start_charge, start_tar_node, start_cur_node, data['num_nodes'])

    if state > data['num_states']:
        logger.error(f'Error in test policy: state = {state} > num_states')
        return

    # Generate path from policy and output the result
    logger.info('test policy')

    path_states, path_nodes = path_from_policy(
        state, P, pi, data['num_nodes'],
        data['max_actions'],
        params['test']['max_iter'],
        params['test']['static_test'])

    logger.info('total path')

    for path_state in path_states:
        charge, tar_node, cur_node = decode_state(path_state, data['num_nodes'])
        logger.info(f'step > \t(charge: {charge}, tar_node: {tar_node}, cur_node: {cur_node})')

    # -----------------------------------------------------------
    # Plot Policy
    # -----------------------------------------------------------

    valid, missing = files_available(
        data_out, ['coordinates.npy', 'charger_ids.npy', 'edgelist_from.npy', 'edgelist_to.npy'])

    if not valid:
        logger.info(f'Missing files: {missing}.')
        return

    coordinates = np.load(os.path.join(data_out, 'coordinates.npy'))
    charger_ids = np.load(os.path.join(data_out, 'charger_ids.npy'))
    edgelist_from = np.load(os.path.join(data_out, 'edgelist_from.npy'))
    edgelist_to = np.load(os.path.join(data_out, 'edgelist_to.npy'))

    if plot:
        # Basic graph of network
        plot_graph('Test Policy', coordinates, edgelist_from, edgelist_to, False)

        # Path locations
        plt.plot(coordinates[0, path_nodes],
                 coordinates[1, path_nodes], 'go-')

        # Charger locations
        plt.plot(coordinates[0, charger_ids],
                 coordinates[1, charger_ids], 'bs')

        # Start location
        plt.plot(coordinates[0, start_cur_node],
                 coordinates[1, start_cur_node], 'gD')

        # Target location
        plt.plot(coordinates[0, start_tar_node],
                 coordinates[1, start_tar_node], 'rD')

        plt.legend(['policy path', 'charger node', 'start node', 'target node'])

        if show_fig:
            plt.show()
        else:
            plt.savefig(os.path.join(data_out, 'network_graph_policy.png'), dpi=300)


if __name__ == '__main__':

    # Generator arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Create plot of full graph')
    parser.add_argument('--force', action='store_true',
                        help='Force updating the policy even if one already exists')
    parser.add_argument('--show_fig', action='store_true', help='Show figure of full graph')
    parser.add_argument('-c', '--config', type=str, help='Specify yaml config file', default='')
    parser.add_argument('-d_out', '--data_out', type=str,
                        help='Specify folder for stored parameters and data', default='')
    args = parser.parse_args()

    # Measure complete time of main
    t0 = time.time()
    main(args.plot, args.show_fig, args.force, args.config, args.data_out)
    dt = time.time() - t0

    # Print time
    if dt < 60:
        logger.info(f'done in {dt} seconds')
    else:
        logger.info(f'done in {dt/60.0} minutes')
