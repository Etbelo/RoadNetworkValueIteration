import time
import argparse
import os
from tracemalloc import start
from unicodedata import numeric
from matplotlib.pyplot import quiver

import numpy as np
from scipy.sparse import csr_matrix
import pickle
import yaml

from backend.backend_cpp import *
from backend.helpers import *


logger = logging.Logger('python')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(name)s] (%(levelname)s) %(message)s'))

logger.addHandler(ch)


def main(plot, show_fig, force, config_file, data_out):

    logger.info('running')

    # -----------------------------------------------------------
    # Read Files and Parameters
    # -----------------------------------------------------------

    params = {}

    with open(config_file, 'rb') as file:
        try:
            params = yaml.safe_load(file)
        except Exception as e:
            logger.error(f'Parameter exception: {e}')

    logger.info(f'params: {params}')

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
    # Evaluate MDP -> Generate Policy
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
    # Test Policy -> Plot Generated Policy
    # -----------------------------------------------------------

    # Continue if parameters and files are valid
    if not plot:
        return

    valid, missing = files_available(
        data_out, ['coordinates.npy', 'charger_ids.npy', 'edgelist_from.npy', 'edgelist_to.npy'])

    if not valid:
        logger.info(f'Missing files: {missing}.')
        return

    # Load additionally required files
    coordinates = np.load(os.path.join(data_out, 'coordinates.npy'))
    charger_ids = np.load(os.path.join(data_out, 'charger_ids.npy'))
    edgelist_from = np.load(os.path.join(data_out, 'edgelist_from.npy'))
    edgelist_to = np.load(os.path.join(data_out, 'edgelist_to.npy'))

    # Test policy from state picked from console and plot directly
    if params['test']['pick_state']:
        # Pick and validate start charge from console
        max_charge = params['generate']['num_charges'] - 1
        start_charge = input(f'\nPick start charge (max: {max_charge}): ')

        if len(start_charge) == 0:
            start_charge = max_charge
        elif not start_charge.isnumeric():
            logger.error(f'Start charge needs to be a numeric value: {start_charge}')
            return

        start_charge = int(start_charge)

        if start_charge > max_charge:
            logger.error(f'Start charge is larger than the maximum: {start_charge} >= {max_charge}')
            return

        # Pick start and target nodes from plot
        logger.info('Pick start and end node by clicking on the plot!')

        # Plot basic graph of network
        fig = plot_graph('Test Policy', coordinates, edgelist_from, edgelist_to)

        # Plot charger locations
        plt.plot(coordinates[0, charger_ids],
                 coordinates[1, charger_ids], 'bs')

        plt.legend(['charger node'])

        # Global variables for on_click handler
        global start_cur_node,  start_tar_node, iter

        start_cur_node = 0
        start_tar_node = 0
        iter = 0

        def onclick(event):
            '''! Callback function when mouse is clicked on plot.

            @param event Mouse event providing coordinate data
            '''

            global start_cur_node, start_tar_node, iter

            iter += 1

            ref_point = np.array([event.xdata, event.ydata])
            node = find_closest_node(coordinates, ref_point)

            # Start node selected
            if iter == 1:
                start_cur_node = node
                plt.plot(coordinates[0, start_cur_node], coordinates[1, start_cur_node], 'gD')
                plt.legend(['charger node', f'start_node={start_cur_node}'])
                plt.draw()

            # Goal node selected
            if iter == 2:
                start_tar_node = node
                plt.plot(coordinates[0, start_tar_node], coordinates[1, start_tar_node], 'rD')
                plt.legend(['charger node', f'start_node={start_cur_node}',
                            f'target_node={start_tar_node}'])
                plt.draw()
                fig.canvas.mpl_disconnect(cid)

                # Test and plot policy
                test_policy(start_charge, start_tar_node, start_cur_node, data, params,
                            coordinates, charger_ids, data_out, P, pi)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    else:
        # Test policy from provided state
        start_charge = params['test']['start_charge']
        start_tar_node = params['test']['target_node']
        start_cur_node = params['test']['start_node']

        # Plot basic graph of network
        plot_graph('Test Policy', coordinates, edgelist_from, edgelist_to)

        # Test and plot policy
        test_policy(start_charge, start_tar_node, start_cur_node, data,
                    params, coordinates, charger_ids, data_out, P, pi)

        if show_fig:
            plt.show()


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
