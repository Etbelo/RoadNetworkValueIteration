import argparse
import os
import time
import logging

import numpy as np
import pandas as pd
import yaml
import pickle

from backend.backend_cpp import *
from backend.helpers import *
from backend.visualization import *

logger = logging.Logger('python')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(name)s] (%(levelname)s) %(message)s'))

logger.addHandler(ch)


def main(plot, show_fig, config_file, data_out, data_in):

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

    logger.info(f'params: {params}')

    # -----------------------------------------------------------
    # Read Dataset
    # -----------------------------------------------------------

    logger.info('read dataset')

    valid, missing = files_available(
        data_in, [os.path.join(params['generate']['dataset'],
                               'nodes.csv'),
                  os.path.join(params['generate']['dataset'],
                               'edges.csv')])

    if not valid:
        logger.error(f'Missing files: {missing}')
        return

    # Read nodes.csv
    nodes_df = pd.read_csv(
        os.path.join('data', params['generate']['dataset'],
                     'nodes.csv'),
        delimiter=' ')

    lat = nodes_df['latitude'].to_numpy()
    long = nodes_df['longitude'].to_numpy()
    coordinates = np.vstack((long, lat))
    num_nodes = lat.size

    # Read edges.csv
    edges_df = pd.read_csv(os.path.join('data', params['generate']['dataset'],
                                        'edges.csv'), delimiter=' ')

    T_row = edges_df['start_id'].to_numpy()
    T_col = edges_df['end_id'].to_numpy()
    T_dist = edges_df['l2_distance'].to_numpy()

    # Read chargers.csv
    if not params['generate']['random_chargers']:

        valid, missing = files_available(
            data_in, [os.path.join(params['generate']['dataset'], 'chargers.csv')])

        if valid:
            chargers_df = pd.read_csv(os.path.join(
                'data', params['generate']['dataset'],
                'chargers.csv'),
                delimiter=' ')

            charger_ids = chargers_df['node_id'].to_numpy()
            chargers = np.zeros(num_nodes, dtype=bool)
            chargers[charger_ids] = True
        else:
            logger.error(f'chargers not found in dataset')

    # Create bi-directional graph adjecency matrix
    T = get_transition_matrix(
        num_nodes, T_row, T_col, T_dist)

    if params['generate']['compress']:
        logger.info('compress dataset')

        # Compress dataset by removing degree 2 nodes from transition graph
        T, T_row, T_col, valid, new_nodes, num_nodes = compress_csr_graph(T)
        coordinates = coordinates[:, valid]
        T_dist = T.sum(0)

    # -----------------------------------------------------------
    # Pre-computation
    # -----------------------------------------------------------

    # Generate random charger locations
    if params['generate']['random_chargers'] or 'chargers' not in locals():
        logger.info('generate random chargers')

        chargers, charger_ids = get_random_chargers(
            num_nodes, params['generate']['num_chargers'])

    # Maximum number of intersections/actions
    max_u, max_node = get_max_u(T)

    # Max/min edge distances
    max_dist = np.max(T_dist)
    min_dist = np.min(T_dist)

    num_states = (num_nodes ** 2) * params['generate']['num_charges']

    # -----------------------------------------------------------
    # Generate MDP
    # -----------------------------------------------------------

    logger.info('generate mdp')

    generate_mdp(
        chargers, T, num_nodes, min_dist, max_dist, max_u, data_out,
        params['generate']['num_charges'],
        params['generate']['max_charge_cost'],
        params['generate']['direct_charge'],
        params['generate']['p_travel'])

    # -----------------------------------------------------------
    # Save data
    # -----------------------------------------------------------

    logger.info('save data')

    p_params = {'p_shape': (max_u * num_states, num_states),
                'num_nodes': num_nodes, 'num_states': num_states, 'max_actions': max_u}

    pickle.dump(p_params, open(os.path.join(data_out, 'p_params.p'), "wb"))

    np.save(os.path.join(data_out, 'charger_ids'), charger_ids)
    np.save(os.path.join(data_out, 'coordinates'), coordinates)
    np.save(os.path.join(data_out, 'edgelist_from'), T_row)
    np.save(os.path.join(data_out, 'edgelist_to'), T_col)

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------

    if plot:
        logger.info('plotting')

        # Plot compressed dataset
        plot_graph('Network Graph',
                   coordinates, T_row, T_col, show=False)

        # Charger locations
        plt.plot(coordinates[0, charger_ids],
                 coordinates[1, charger_ids], 'bs')

        # Maximum number of intersections
        plt.plot(coordinates[0, max_node], coordinates[1, max_node], 'ro')

        # Legend for compressed graph
        plt.legend(['Random Chargers', 'Highest Intersection'])

        if show_fig:
            plt.show()
        else:
            plt.savefig(os.path.join(data_out, 'network_graph.png'), dpi=300)


if __name__ == '__main__':

    # Generator arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Create plot of full graph')
    parser.add_argument(
        '--show_fig', action='store_true',
        help='Show figures of full graph; Otherwise, images of all figures are stored to data_out')
    parser.add_argument('-c', '--config', type=str,
                        help='Specify path to yaml configuration file', default='')
    parser.add_argument('-d_out', '--data_out', type=str,
                        help='Specify folder path to stored parameters and data', default='')
    parser.add_argument('-d_in', '--data_in', type=str,
                        help='Specify folder path to datasets', default='')
    args = parser.parse_args()

    # Measure complete time of main
    t0 = time.time()
    main(args.plot, args.show_fig, args.config, args.data_out, args.data_in)
    dt = time.time() - t0

    # Print time
    if dt < 60:
        logger.info(f'done in {dt} seconds')
    else:
        logger.info(f'Done in {dt/60.0} minutes')
