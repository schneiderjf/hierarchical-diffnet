import pandas as pd
import re


# Define functions
def get_nodename(x):
    try:
        matchobj1 = re.search('://(.*?)/', str(x))
        return matchobj1.group(1)
    except:
        return None


def filter_data(x):
    try:
        matchobj1 = re.match('2008-', str(x))
        matchobj1.group(0)
        return x
    except:
        return None


def assign_content_cluster(x):
    pass


class Preprocessing():
    def __init__(self, path):
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
        # parse url
        data['node'] = data['<Url>'].apply(lambda x: get_nodename(x))
        # fill na
        data = data.fillna(method='ffill')
        # filter out top nodes and replace with an id:
        nodes = data[data['<UrlTy>'] == 'M'].groupby('node').count().sort_values(
            ascending=False, by='<Url>').reset_index()
        news = nodes.head(num_nodes).node.values
        data = data[data.node.isin(news[1:])]
        labels = dict()

        j = 0
        for i in news[1:]:
            labels[i] = j
            j += 1
        data['node'] = data['node'].replace(labels)

        # filter data and type transformation, calculate the value
        data['<Tm>'] = data['<Tm>'].apply(lambda x: filter_data(x))
        data['<Tm>'] = pd.to_datetime(data['<Tm>'])
        data['diff'] = data.groupby('Unnamed: 1')['<Tm>'].transform(
            lambda x: (x - min(x)))
        data['diff'] = data['diff'].apply(lambda x: x.total_seconds())
        data = data.drop(['<Url>', '<UrlTy>', '<Fq>', 'Unnamed: 0', '<Tm>'],
                         axis=1)
        data = data[data['diff'].notnull()]
        data.columns = ['cascade_id', 'node_id', 'hours_till_start']
        data['hours_till_start'] = data['hours_till_start'] / 3600
        data = data.sort_values('hours_till_start')
        data = data.drop_duplicates(['cascade_id', 'node_id'])
        data['hours_till_start'] = data.groupby('cascade_id')[
            'hours_till_start'].transform(
            lambda x: (x - min(x)))

        self.data = data
        self.labels = labels

        return None