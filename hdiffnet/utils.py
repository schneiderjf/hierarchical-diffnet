import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score
import time
from IPython import display
from tensorflow_probability import edward2 as ed



def transform_full_to_sparse(data, topics=False):
    a = data.groupby('cascade_id')['node'].apply(list)
    b = data.groupby('cascade_id')['t'].apply(list)
    if topics:
        topics = data.groupby('cascade_id')['polarity',
                                            'polarity2'].mean().\
            values.astype(np.float32)
    result = []
    for i in range(a.shape[0]):
        res = list(itertools.chain(*zip(a.iloc[i], b.iloc[i])))
        result.append(res)
    return result, topics


def buildGraph(alpha):
    graph = nx.from_numpy_matrix(alpha)
    layout = nx.spring_layout(graph)
    weights = [graph[u][v]['weight'] * 5 / alpha.max() for u, v in
               graph.edges()]
    labels = {node: str(node) for node in graph.nodes()}

    return graph, layout, weights, labels


def drawEmptyGraph(graph, layout, labels):
    fig, ax = plt.subplots(figsize=(8, 7))

    nx.draw_networkx_nodes(graph,
                           layout,
                           node_color='r',
                           alpha=1,
                           ax=ax)
    nx.draw_networkx_labels(graph,
                            layout,
                            labels=labels, font_color="white", ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])


def drawWeightedGraph(graph, layout, weights, labels):
    nx.draw(graph, layout, edges=graph.edges, width=weights, labels=labels,
            font_color="white")


def sampleCascade(alpha, T):
    sess = tf.Session()
    alpha_tf = tf.convert_to_tensor(alpha, dtype=tf.float32)
    tau = ed.Exponential(alpha_tf)
    cascade = sess.run(build_cascade(tau, 0, T))

    return cascade


def build_cascade(time, seed, T):
    # Store number of nodes
    n = time.shape[0]

    # Transpose times and reduce minimum
    times_T = tf.minimum(tf.transpose(time), T)

    # Initialize transmission times to be max time except for seed node
    transmission = tf.ones(n) * T
    transmission = tf.subtract(transmission, tf.one_hot(seed, n) * T)

    # Continually update transmissions
    for _ in range(n):
        # Tile transmission
        transmission_tiled = tf.reshape(tf.tile(transmission, [n]), [n, n])

        # Add transposed times and tiled transmissions
        potential_transmission = tf.add(transmission_tiled, times_T)

        # Find minimum path from all new
        potential_transmission_row = tf.reduce_min(potential_transmission,
                                                   reduction_indices=[1])

        # Concatenate previous transmission and potential new transmission
        potential_transmission_stack = tf.stack([transmission,
                                                 potential_transmission_row], axis=0)

        # Take the minimum of the original transmission and the potential new transmission
        transmission = tf.reduce_min(potential_transmission_stack, reduction_indices=[0])

    return transmission


def printCascade(cascade, T):
    print("node\t time")
    print("----\t ----")

    cascade_order = cascade.argsort().tolist()
    i = 0

    while i <= len(cascade_order) - 1 and cascade[cascade_order[i]] < T:
        print('{:4d}\t {:0.2f}'.format(cascade_order[i], cascade[cascade_order[i]]))
        i += 1


def drawNetworkProp(graph, layout, labels, cascade, T):
    cascade_order = cascade.argsort().tolist()

    fig, ax = plt.subplots(figsize=(8, 7))

    for num in range(len(graph.nodes)):
        if cascade[cascade_order[num]] == T: break
        time.sleep(1)

        # Draw infected nodes
        inf = nx.draw_networkx_nodes(graph,
                                     layout,
                                     node_color='r',
                                     nodelist=cascade_order[:num + 1],
                                     alpha=1,
                                     ax=ax)

        # Draw uninfected nodes
        uninf = nx.draw_networkx_nodes(graph,
                                       layout,
                                       edge_color='b',
                                       node_color='w',
                                       nodelist=cascade_order[num + 1:],
                                       alpha=1,
                                       ax=ax)
        try:
            uninf.set_edgecolor("black")
        except:
            None

        # Draw node
        nx.draw_networkx_labels(graph,
                                layout,
                                labels=labels, font_color="white", ax=ax)

        # Scale plot ax
        ax.set_xticks([])
        ax.set_yticks([])

        # Redraw cascade
        display.clear_output(wait=True)
        display.display(fig)

    display.clear_output(wait=True)
    return printCascade(cascade, T)

def get_seed_set(r):
    results = []
    for i in r:
        results.append(i[0])
    return results


def get_max_times(r):
    results = []
    for i in r:
        results.append(max(i))
    return results


def convert_to_matrix(v, numNodes, maxT):
    # convert the kronecker into the vector format:
    np_cascades = np.multiply(np.ones((len(v), numNodes), np.float32),
                              np.array(maxT).reshape((len(v), 1)))
    for row, cascade in enumerate(v):
        c_nodes = [int(cascade[i * 2]) for i in range(len(cascade) // 2)]
        c_times = [cascade[i * 2 + 1] for i in range(len(cascade) // 2)]

        for col in range(len(c_nodes)):
            np_cascades[row][c_nodes[col]] = c_times[col]

    return np_cascades


def evaluate(test_set, test_cascades, benchmark_cascades, times):
    mse1 = np.mean((test_set - test_cascades) ** 2)
    mse2 = np.mean((test_set - benchmark_cascades) ** 2)
    mae1 = np.mean(np.absolute(test_set - test_cascades))
    mae2 = np.mean(np.absolute(test_set - benchmark_cascades))
    kendall1 = kendalltau(np.argsort(test_set),
                          np.argsort(test_cascades)).correlation
    kendall2 = kendalltau(np.argsort(test_set),
                          np.argsort(benchmark_cascades)).correlation

    b = np.equal(test_set,
                 np.tile(np.array(times).reshape((test_set.shape[0], 1)),
                         reps=(1, test_set.shape[1])))
    s1 = np.where(b, np.zeros(test_set.shape), np.ones(test_set.shape))
    b = np.isclose(test_cascades,
                   np.tile(test_cascades.max(axis=1).
                           reshape((test_set.shape[0], 1)),
                           reps=(1, test_set.shape[1])),
                   atol=1)
    s2 = np.where(b, np.zeros(test_set.shape), np.ones(test_set.shape))
    b = np.isclose(benchmark_cascades,
                   np.tile(benchmark_cascades.max(axis=1).
                           reshape((test_set.shape[0], 1)),
                           reps=(1, test_set.shape[1])),
                   atol=1)
    s3 = np.where(b, np.zeros(test_set.shape), np.ones(test_set.shape))

    prec1 = average_precision_score(s1.flatten(), s2.flatten())
    prec2 = average_precision_score(s1.flatten(), s3.flatten())

    print("metric\t set\t \t value")
    print("_______|_____________|______________________")
    print("MSE\t benchmark\t" + str(round(mse2)))
    print("MSE\t holdout\t" + str(round(mse1)))
    print("MAE\t benchmark\t" + str(round(mae2)))
    print("MAE\t holdout\t" + str(round(mae1)))
    print("KRCC \t benchmark\t" + str(round(kendall2, 3)))
    print("KRCC \t holdout\t" + str(round(kendall1, 3)))
    print("Prec\t benchmark\t" + str(round(prec2)))
    print("Prec\t holdout\t" + str(round(prec1)))

    return None
