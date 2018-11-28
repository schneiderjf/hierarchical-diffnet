import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.metrics import average_precision_score
import time
from IPython import display


def transform_full_to_sparse(data, topics=False):
    """
    transforms a matrix cascades into a sparse cascade format
    :param data: pd.dataframe | from the preprocessing class
    :param topics: bool | indicates whether a topic vector should
    :return: list of list, np.array
    """
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


def buildGraph(alpha=None):
    """
    Generate graphs
    :param alpha:
    :return:
    """
    if alpha is None:
        alpha = defaultAlpha()

    graph = nx.from_numpy_matrix(alpha)
    layout = nx.spring_layout(graph)
    weights = [graph[u][v]['weight'] * 5 / alpha.max() for u, v in
               graph.edges()]
    labels = {node: str(node) for node in graph.nodes()}

    return graph, layout, weights, labels


def drawEmptyGraph(graph, layout, labels):
    """
    generate graphs
    :param graph:
    :param layout:
    :param labels:
    :return:
    """
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
    """
    generate graphs
    :param graph:
    :param layout:
    :param weights:
    :param labels:
    :return:
    """
    nx.draw(graph, layout, edges=graph.edges, width=weights, labels=labels,
            font_color="white")


def defaultAlpha():
    """
    generate graphs
    :return:
    """
    return np.array([[0, .1, 0, 0, 0, .2, 0, 0, 0, 0],
                     [0, 0, .5, 0, 0, 0, 0, .25, 0, 0],
                     [0, 0, 0, .4, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, .5, 0, 0, .1, 0, 0],
                     [0, 0, 0, 0, 0, .3, 0, 0, .2, 0],
                     [0, 0, 0, 0, 0, 0, .7, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, .6, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, .3],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)


def printCascade(cascade, T):
    """
    print a cascade
    :param cascade:
    :param T:
    :return:
    """
    print("node\t time")
    print("----\t ----")

    cascade_order = cascade.argsort().tolist()
    i = 0

    while i <= len(cascade_order) - 1 and cascade[cascade_order[i]] < T:
        print('{:4d}\t {:0.2f}'.format(cascade_order[i],
                                       cascade[cascade_order[i]]))
        i += 1


def drawNetworkProp(graph, layout, labels, cascade, T):
    """
    generate graphs
    :param graph:
    :param layout:
    :param labels:
    :param cascade:
    :param T:
    :return:
    """
    cascade_order = cascade.argsort().tolist()

    fig, ax = plt.subplots(figsize=(8, 7))

    for num in range(len(graph.nodes)):
        if cascade[cascade_order[num]] == T:
            break
        time.sleep(1)

        nx.draw_networkx_nodes(graph,
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
        if uninf is not None:
            uninf.set_edgecolor("black")

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
    """
    evaluates several metrcis between the test cascades and the model generated
    cascades as well as a benchmark model generated set of cascades. Times indi
    cates the time horizon for each cascade. All arrays have to be same shaped
    (except the time array)
    :param test_set: np.array
    :param test_cascades: np.array
    :param benchmark_cascades: np.array
    :param times: np.array
    :return: None
    """
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
    print("MSE\t model  \t" + str(round(mse1)))
    print("MAE\t benchmark\t" + str(round(mae2)))
    print("MAE\t model  \t" + str(round(mae1)))
    print("KRCC \t benchmark\t" + str(round(kendall2, 3)))
    print("KRCC \t model  \t" + str(round(kendall1, 3)))
    print("Prec\t benchmark\t" + str(round(prec2, 3)))
    print("Prec\t model  \t" + str(round(prec1, 3)))

    return None
