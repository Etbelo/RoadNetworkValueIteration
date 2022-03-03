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

logging.basicConfig(level=logging.INFO, format='[python] %(message)s')


def main(plot, show_fig, config_file, data_out):

    print('')
    logging.info('running')

    # -----------------------------------------------------------
    # Read Parameters
    # -----------------------------------------------------------

    params = {}

    with open(config_file, 'rb') as file:
        try:
            params = yaml.safe_load(file)
        except Exception as e:
            print(f'Parameter exception: {e}')

    # -----------------------------------------------------------
    # Read Dataset
    # -----------------------------------------------------------

    logging.info('read dataset')

    # long = np.array([-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
    # lat = np.array([0.0, 0.0, 1.0, 2.0, -1.0, -2.0, -1.0, -2.0, 0.0, 0.0])

    nodes_df = pd.read_csv('data/nodes.csv', delimiter=' ')

    lat = nodes_df['latitude'].to_numpy()
    long = nodes_df['longitude'].to_numpy()
    coordinates = np.vstack((long, lat))
    num_nodes = lat.size

    edges_df = pd.read_csv('data/edges.csv', delimiter=' ')

    start_ids = edges_df['start_id'].to_numpy()
    end_ids = edges_df['end_id'].to_numpy()
    distances = edges_df['l2_distance'].to_numpy()

    # start_ids = np.array([0, 1, 2, 1, 4, 4, 6, 6, 8, 2])
    # end_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 8])
    # distances = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 0.5, 1.0, 2.0])

    # Create bi-directional graph adjecency matrix
    T = get_transition_matrix(
        num_nodes, start_ids, end_ids, distances)

    logging.info('compress dataset')

    # Compress dataset by removing degree 2 nodes from transition graph
    T_cmp, T_cmp_row, T_cmp_col, valid, num_nodes_cmp = compress_csr_graph(T)
    coordinates_cmp = coordinates[:, valid]
    distances_cmp = T_cmp.sum(0)

    # -----------------------------------------------------------
    # Pre-computation
    # -----------------------------------------------------------

    # Random charger locations
    chargers, charger_ids = get_random_chargers(
        num_nodes_cmp, params['generate']['num_chargers'])

    # Maximum number of intersections/actions
    max_u, max_node = get_max_u(T_cmp)

    # Max/min edge distances
    max_dist = np.max(distances_cmp)
    min_dist = np.min(distances_cmp)

    num_states = num_nodes_cmp ** 2 * params['generate']['num_charges']

    # -----------------------------------------------------------
    # Generate MDP
    # -----------------------------------------------------------

    logging.info('generate mdp')

    generate_mdp(
        chargers, T, num_nodes_cmp, min_dist, max_dist, max_u, data_out,
        params['generate']['num_charges'],
        params['generate']['max_charge'],
        params['generate']['sigma_env'],
        params['generate']['p_travel'],
        params['generate']['n_charge'])

    # -----------------------------------------------------------
    # Save data
    # -----------------------------------------------------------

    logging.info('save data')

    p_params = {'p_shape': (max_u * num_states, num_states),
                'num_nodes': num_nodes_cmp, 'num_states': num_states, 'max_actions': max_u}

    pickle.dump(p_params, open(os.path.join(data_out, 'p_params.p'), "wb"))

    np.save(os.path.join(data_out, 'charger_ids'), charger_ids)
    np.save(os.path.join(data_out, 'coordinates'), coordinates_cmp)
    np.save(os.path.join(data_out, 'edgelist_from'), T_cmp_row)
    np.save(os.path.join(data_out, 'edgelist_to'), T_cmp_col)

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------

    if plot:
        logging.info('plotting')

        # Plot original dataset
        plot_graph('Graph of Road Network', coordinates, start_ids, end_ids, show=False)

        if not show_fig:
            plt.savefig(os.path.join(data_out, 'road_network.png'), dpi=300)

        # Plot compressed dataset
        plot_graph('Graph of Compressed Road Network',
                   coordinates_cmp, T_cmp_row, T_cmp_col, show=False)

        # Charger locations
        plt.plot(coordinates_cmp[0, charger_ids],
                 coordinates_cmp[1, charger_ids], 'bs')

        # Maximum number of intersections
        plt.plot(coordinates_cmp[0, max_node], coordinates_cmp[1, max_node], 'ro')

        # Legend for compressed graph
        plt.legend(['Random Chargers', 'Highest Intersection'])

        if show_fig:
            plt.show()
        else:
            plt.savefig(os.path.join(data_out, 'road_network_cmp.png'), dpi=300)


if __name__ == '__main__':

    # Generator arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help='Create plot of full graph')
    parser.add_argument(
        '--show_fig', action='store_true',
        help='Show figures of full graph; Otherwise, images of all figures are stored to data_out')
    parser.add_argument('-c', '--config', type=str,
                        help='Specify yaml config file', default='config.yaml')
    parser.add_argument('-d', '--data_out', type=str,
                        help='Specify folder for stored parameters and data', default='data_out')
    args = parser.parse_args()

    # Measure complete time of main
    t0 = time.time()
    main(args.plot, args.show_fig, args.config, args.data_out)
    dt = time.time() - t0

    # Print time
    if dt < 60:
        print(f'\nDone in {dt} seconds.', flush=True)
    else:
        print(f'\nDone in {dt/60.0} minutes.', flush=True)
