import matplotlib.pyplot as plt
import numpy as np


def plot_graph(title, coordinates, start_ids, end_ids, show=True):

    connections = np.zeros((start_ids.size, 4), dtype=np.float32)

    for i, (start, end) in enumerate(zip(start_ids, end_ids)):
        connections[i, 0:2] = coordinates[:, start]
        connections[i, 2:4] = coordinates[:, end] - coordinates[:, start]

    fig = plt.figure()
    fig.suptitle(title)

    plt.quiver(connections[:, 0],
               connections[:, 1],
               connections[:, 2],
               connections[:, 3],
               color='black',
               headwidth=1,
               headlength=0,
               linewidth=0.5,
               width=0.001,
               scale_units='xy',
               scale=1.0,
               angles='xy')

    plt.axis('equal')
    plt.xlim([np.min(coordinates[0, :])-1.0, np.max(coordinates[0, :])+1.0])
    plt.ylim([np.min(coordinates[1, :])-1.0, np.max(coordinates[1, :])+1.0])
    plt.xlabel('latitude')
    plt.ylabel('longitude')

    if show:
        plt.show()
