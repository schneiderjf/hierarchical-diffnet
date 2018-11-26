import itertools

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