import matplotlib.pyplot as plt
import networkx as nx
import time
import itertools
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score, precision_recall_curve

def transform_full_to_sparse(data):
    a = data.groupby('cascade_id')['node_id'].apply(list)
    b = data.groupby('cascade_id')['hours_till_start'].apply(list)
    result = []
    for i in range(a.shape[0]):
        res = list(itertools.chain(*zip(a.iloc[i], b.iloc[i])))
        result.append(res)
    return result


def buildGraph(alpha):
    graph = nx.from_numpy_matrix(alpha)
    layout = nx.spring_layout(graph)
    weights = [graph[u][v]['weight'] * 5 / alpha.max() for u, v in
               graph.edges()]
    labels = {node: str(node) for node in graph.nodes()}

    return graph, layout, weights, labels

def drawEmptyGraph(graph, layout, labels):
    nx.draw(graph, layout, edges=graph.edges, labels=labels,  font_color="white")

def drawWeightedGraph(graph, layout, weights, labels):
    nx.draw(graph, layout, edges=graph.edges, width=weights, labels=labels,
            font_color="white")

def drawNetworkProp(graph, layout, labels, cascade, T):
    cascade_order = cascade.argsort().tolist()

    fig, ax = plt.subplots(figsize=(8, 7))

    for num in range(len(graph.nodes)):
        if cascade[cascade_order[num]] == T: break
        time.sleep(1)

        # Draw edges
        nx.draw_networkx_edges(graph, layout, width=1.0, alpha=0.5,
                               labels=True, ax=ax)

        # Draw infected nodes
        inf = nx.draw_networkx_nodes(nx_graph,
                                     layout,
                                     node_color='r',
                                     nodelist=cascade_order[:num + 1],
                                     alpha=1,
                                     ax=ax)

        # Draw uninfected nodes
        uninf = nx.draw_networkx_nodes(nx_graph,
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
        nx.draw_networkx_labels(nx_graph,
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


# squared loss

def evaluate(test_set, test_cascades, benchmark_cascades):
    mse1 = np.mean((test_set - test_cascades) ** 2)
    mse2 = np.mean((test_set - benchmark_cascades) ** 2)
    mae1 = np.mean(np.absolute(test_set - test_cascades))
    mae2 = np.mean(np.absolute(test_set - benchmark_cascades))
    kendall1 = kendalltau(np.argsort(test_set),
                          np.argsort(test_cascades)).correlation
    kendall2 = kendalltau(np.argsort(test_set),
                          np.argsort(benchmark_cascades)).correlation

    b = np.equal(test_set,
                 np.tile(np.array(times).reshape((100, 1)), reps=(1, 200)))
    s1 = np.where(b, np.zeros((100, 200)), np.ones((100, 200)))
    b = np.isclose(test_cascades,
                   np.tile(test_cascades.max(axis=1).reshape((100, 1)),
                           reps=(1, 200)),
                   atol=1)
    s2 = np.where(b, np.zeros((100, 200)), np.ones((100, 200)))
    b = np.isclose(benchmark_cascades,
                   np.tile(benchmark_cascades.max(axis=1).reshape((100, 1)),
                           reps=(1, 200)),
                   atol=1)
    s3 = np.where(b, np.zeros((100, 200)), np.ones((100, 200)))

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