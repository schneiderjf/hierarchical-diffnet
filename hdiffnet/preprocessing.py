import pandas as pd
import re


# Define Helper functions
def get_nodename(x):
    matchobj1 = re.search('://(.*?)/', str(x))
    if matchobj1:
        return matchobj1.group(1)
    else:
        return None


def filter_data(x):
    matchobj1 = re.match('2008-', str(x))
    if matchobj1:
        return matchobj1.group(0)
    else:
        return None


def get_cluster_id(a):
    if str(a[1]) == 'nan' and str(a[2]) == 'nan':
        return a[0]
    else:
        return None


def get_cascade_id(a):
    if not (str(a[1]) == 'nan') and str(a[2]) == 'nan':
        return a[1]
    else:
        return None


def get_polarity(x):
    """
    For a cascade word cloud, it will retrieve its topic in a heuristic way
    :param x:
    :return:
    """
    x = str(x).lower()
    words_politics = ['obama', 'biden', 'joe biden', 'white house', 'congress',
                      'election', 'administration', 'democrat', 'replublican',
                      'democracy',
                      'congressman', 'senat', 'senator', 'household',
                      'midterm', 'supreme court',
                      'politic']
    words_sports = ['baseball', 'field goal', 'football', 'NFL', 'basketball',
                    'player', 'sports',
                    'trainer', 'coach', 'ball']

    for s in words_politics:
        matchobj1 = re.search('(' + str(s) + '?)', x)
        if matchobj1:
            return 0
    for s in words_sports:
        matchobj1 = re.search('(' + str(s) + '?)', x)
        if matchobj1:
            return 1
    else:
        return None


class Preprocessing():
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path,
                                delimiter='\t',
                                skiprows=3)
        self.labels = None

    def preprocess_data(self, num_nodes):
        """
        transforms and cleans the data, so it is in a format that can be used
        :param num_nodes: Number of
        :return: pd.df of the cleaned data
        """
        data = self.data
        data['cluster_id'] = None
        data['cascade_id'] = None
        data = data.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)
        data['node'] = data['<Url>'].apply(lambda x: get_nodename(x))
        data['cluster_id'] = data[['<Fq>', '<UrlTy>', '<Url>']].apply(
            lambda x: get_cluster_id(x), axis=1)
        data['cascade_id'] = data[['<Fq>', '<UrlTy>', '<Url>']].apply(
            lambda x: get_cascade_id(x), axis=1)

        data['cluster_id'] = data['cluster_id'].fillna(method='ffill')
        data['cascade_id'] = data['cascade_id'].fillna(method='ffill')
        nodes = data[data['<UrlTy>'] == 'M']. \
            groupby('node').count(). \
            sort_values(ascending=False, by='<Url>'). \
            reset_index()
        news = nodes.head(num_nodes).node.values
        data = data[data.node.isin(news[1:])]
        labels = dict()

        j = 0
        for i in news[1:]:
            labels[i] = j
            j += 1
        data['node'] = data['node'].replace(labels)
        data['<Tm>'] = pd.to_datetime(data['<Tm>'])
        data['t'] = data.groupby('cascade_id')['<Tm>'].transform(
            lambda x: (x - min(x)))
        data['t'] = data['t'].apply(lambda x: x.total_seconds())
        data = data.drop(['<Url>', '<UrlTy>', '<Fq>', '<Tm>'],
                         axis=1)
        data = data.drop_duplicates(['cascade_id', 'node'])
        data['t'] = data['t'] / 3600
        data = data.groupby('cascade_id').filter(lambda x: len(x) > 4)
        data = data.sort_values('t')

        self.data = data
        self.labels = labels
        self.cascade_ids = data.cascade_id.unique()

        return None

    def add_polarity(self):
        """
        adds polarity data to each cascade id, by using a heuristic appraoch
        :return:
        """
        data = pd.read_csv(self.path,
                           delimiter='\t', skiprows=3)
        v = self.cascade_ids

        data['cluster_id'] = None
        data['cascade_id'] = None
        data = data.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)
        data['node'] = data['<Url>'].apply(lambda x: get_nodename(x))
        data['cluster_id'] = data[['<Fq>', '<UrlTy>', '<Url>']].apply(
            lambda x: get_cluster_id(x), axis=1)
        data['cascade_id'] = data[['<Fq>', '<UrlTy>', '<Url>']].apply(
            lambda x: get_cascade_id(x), axis=1)

        data['cluster_id'] = data['cluster_id'].fillna(method='ffill')
        data['cascade_id'] = data['cascade_id'].fillna(method='ffill')

        words = data[data.cascade_id.isin(v) & data['<Url>'].isna()]
        bofw = words.groupby('cluster_id')['<Fq>'].apply(lambda x: " ".join(x))
        clusters = bofw.apply(lambda x: get_polarity(x)).dropna()

        data = self.data
        data = data[data.cluster_id.isin(clusters.index)]
        data['polarity'] = data.cluster_id.astype(str).replace(
            clusters.to_dict())
        data['polarity2'] = (data['polarity'] - 1) ** 2

        self.data = data

        return None
