import tensorflow as tf
import numpy as np


def infectedCascade(cascade, N, T=10):
    inf = np.zeros((N, N))

    c_nodes = [int(cascade[i * 2]) for i in range(len(cascade) // 2)]
    c_times = [cascade[i * 2 + 1] for i in range(len(cascade) // 2)]

    for i in range(len(c_nodes)):
        for j in range(i):
            if cascade[j] < T:
                inf[(c_nodes[i], c_nodes[j])] = c_times[i] - c_times[j]

    return tf.convert_to_tensor(inf)


def uninfectedCascade(cascade, N, T=6):
    nodes = {s for s in range(N)}
    uninf = np.zeros((N, N))

    c_nodes = [int(cascade[i * 2]) for i in range(len(cascade) // 2)]
    c_times = [cascade[i * 2 + 1] for i in range(len(cascade) // 2)]

    for i in range(len(c_nodes)):
        for j in (nodes - set(c_nodes)):
            uninf[c_nodes[i], j] = T - c_times[i]

    return tf.convert_to_tensor(uninf)


def genInfectedTensor(v, numNodes, T):
    tf_infected = None
    for cascade in v:
        if tf_infected is None:
            tf_infected = tf.expand_dims(infectedCascade(cascade,
                                                         numNodes,
                                                         T), 0)
        else:
            tf_infected = tf.concat([tf_infected,
                                     tf.expand_dims(infectedCascade(cascade,
                                                                    numNodes,
                                                                    T), 0)],
                                    axis=0)
    return tf_infected


def genUninfectedTensor(v, numNodes, T):
    tf_uninfected = None
    for cascade in v:
        if tf_uninfected is None:
            tf_uninfected = tf.expand_dims(uninfectedCascade(cascade,
                                                             numNodes,
                                                             T), 0)
        else:
            tf_uninfected = tf.concat([tf_uninfected,
                                       tf.expand_dims(
                                           uninfectedCascade(cascade,
                                                             numNodes,
                                                             T), 0)],
                                      axis=0)
    return tf_uninfected


def f_psi_3(alpha_tensor, infected):
    infected_sign = tf.cast(tf.sign(infected), tf.float32)

    # Row sum infected
    alpha_tensor_row = tf.reduce_sum(tf.multiply(infected_sign, alpha_tensor),
                                     axis=1)

    # Add 1 to 0 entries so log(1)=0
    alpha_tensor_row_zeros = -tf.cast(tf.sign(alpha_tensor_row),
                                      tf.float32) + 1
    return tf.reduce_sum(
        tf.log(tf.add(alpha_tensor_row, alpha_tensor_row_zeros)))


def f_psi_2(alpha_tensor, uninfected):
    return -tf.reduce_sum(tf.multiply(tf.transpose(alpha_tensor),
                                      tf.cast(uninfected, dtype=tf.float32)))


def f_psi_1(alpha_tensor, infected):
    return -tf.reduce_sum(tf.multiply(alpha_tensor,
                                      tf.cast(infected,
                                              dtype=tf.float32)))


def gamma_prior(alpha_tensor):
    return tf.multiply(-tf.ones(alpha_tensor.shape), alpha_tensor)


def f_psi_1_t(rate_intercept,
              rate_affinity,
              infected,
              topic,
              numNodes,
              numTopics=2):
    rate_affinity_rshp = tf.add(tf.reshape(rate_affinity, (numNodes * numNodes,
                                                           numTopics)), .001)
    topic_rshp = tf.reshape(topic, (numTopics, 1))
    affinity = tf.reshape(tf.matmul(rate_affinity_rshp, topic_rshp),
                          (numNodes, numNodes))

    alpha_tensor = tf.add(tf.nn.relu(tf.add(rate_intercept, affinity)), .001)
    return -tf.reduce_sum(
        tf.multiply(alpha_tensor, tf.cast(infected, dtype=tf.float32)))


def f_psi_2_t(rate_intercept,
              rate_affinity,
              uninfected,
              topic,
              numNodes,
              numTopics=2):
    rate_affinity_rshp = tf.add(tf.reshape(rate_affinity,
                                           (numNodes * numNodes, numTopics)),
                                .001)
    topic_rshp = tf.reshape(topic, (numTopics, 1))
    affinity = tf.reshape(tf.matmul(rate_affinity_rshp,
                                    topic_rshp), (numNodes, numNodes))

    alpha_tensor = tf.add(tf.nn.relu(tf.add(rate_intercept, affinity)), .001)

    return -tf.reduce_sum(tf.multiply(tf.transpose(alpha_tensor),
                                      tf.cast(uninfected, dtype=tf.float32)))


def f_psi_3_t(rate_intercept,
              rate_affinity,
              infected,
              topic,
              numNodes,
              numTopics=2):
    rate_affinity_rshp = tf.add(tf.reshape(rate_affinity,
                                           (numNodes * numNodes,
                                            numTopics)), .001)
    topic_rshp = tf.reshape(topic, (numTopics, 1))
    alpha_tensor = tf.add(tf.nn.relu(tf.add(rate_intercept, tf.reshape(
        tf.matmul(rate_affinity_rshp, topic_rshp),
        (numNodes, numNodes)))), .001)

    infected_sign = tf.cast(tf.sign(infected), tf.float32)

    # Row sum infected
    alpha_tensor_row = tf.reduce_sum(tf.multiply(infected_sign, alpha_tensor),
                                     axis=1)

    # Add 1 to 0 entries so log(1)=0
    alpha_tensor_row_zeros = -tf.cast(tf.sign(alpha_tensor_row),
                                      tf.float32) + 1

    return tf.reduce_sum(
        tf.log(tf.add(alpha_tensor_row, alpha_tensor_row_zeros)))


def gamma_prior_t(rate_intercept, rate_affinity,
                  theta_topics, numNodes, numTopics):
    rate_affinity_rshp = tf.add(tf.reshape(rate_affinity, (numNodes * numNodes,
                                                           numTopics)), .001)

    return -tf.add(tf.nn.relu(rate_intercept + tf.reshape(
        tf.matmul(rate_affinity_rshp, theta_topics), (numNodes,
                                                      numNodes))), .001)


def evaluateAlpha(rate_intercept, rate_affinity, topic, numNodes, numTopics):
    rate_affinity_rshp = tf.add(tf.reshape(rate_affinity, (numNodes * numNodes,
                                                           numTopics)), .001)
    topic_rshp = tf.convert_to_tensor(np.reshape(topic, (numTopics, 1)),
                                      dtype=tf.float32)
    alpha_tensor = tf.add(tf.nn.relu(tf.add(rate_intercept, tf.reshape(
        tf.matmul(rate_affinity_rshp, topic_rshp), (numNodes,
                                                    numNodes)))), .001)

    negative_I = 1 - tf.eye(numNodes)

    return tf.multiply(alpha_tensor, negative_I)


class ProbabilityModel():
    """
    The main class for the log-probability model
    """
    def __init__(self, data, numNodes, T, topics=None):
        """

        :param data: a list of list | for the cascades
        :param numNodes: int | the number of nodes in the graph
        :param T: np.array | the time horizon
        :param topics: np.array
        """
        self.data = data
        self.sess = tf.Session()
        self.a = None
        self.numNodes = numNodes
        self.T = T
        self.topics = topics

    def map_estimate_BFGS(self,
                          max_iter=1000,
                          initialize=True,
                          batch_size=None):
        """
        Runs BFGS on the data
        :param max_iter: int | number of max iterations for BFGS
        :param initialize: bool | set to True if alpha values should be
        initialized
        :param batch_size: int | needed for batch update
        :return: alpha
        """
        sess = self.sess
        max_iter = max_iter
        if initialize:
            self.batch = 0

        if batch_size is None:
            Inf = genInfectedTensor(self.data, self.numNodes, self.T)
            U = genUninfectedTensor(self.data, self.numNodes, self.T)
        else:
            Inf = genInfectedTensor(self.data[self.batch:self.batch +
                                    batch_size],
                                    self.numNodes, self.T)
            U = genUninfectedTensor(self.data[self.batch:self.batch +
                                    batch_size],
                                    self.numNodes, self.T)
            self.batch += batch_size

        U_ph = tf.placeholder(tf.float32, U.shape)
        I_ph = tf.placeholder(tf.float32, Inf.shape)
        if initialize:
            B = tf.Variable(tf.random_uniform(U.shape[1:]), dtype=tf.float32)
        else:
            B = tf.Variable(self.a, dtype=tf.float32)
        alpha_tensor = tf.nn.relu(B)

        psi_1 = tf.map_fn(lambda x: f_psi_1(tf.transpose(alpha_tensor), x),
                          I_ph, dtype=tf.float32)
        psi_2 = tf.map_fn(lambda x: f_psi_2(tf.transpose(alpha_tensor), x),
                          U_ph, dtype=tf.float32)
        psi_3 = tf.map_fn(lambda x: f_psi_3(tf.transpose(alpha_tensor), x),
                          I_ph, dtype=tf.float32)
        prior = gamma_prior(alpha_tensor)

        log_p = -(tf.reduce_sum(psi_1) +
                  tf.reduce_sum(psi_2) +
                  tf.reduce_sum(psi_3) +
                  tf.reduce_sum(prior))

        feed_dict = {U_ph: U.eval(session=sess),
                     I_ph: Inf.eval(session=sess)}

        optimizer = tf.contrib.opt.\
            ScipyOptimizerInterface(log_p,
                                    method='L-BFGS-B',
                                    options={'maxiter': max_iter})
        if initialize:
            model = tf.global_variables_initializer()

        sess.run(model)
        optimizer.minimize(sess, feed_dict=feed_dict)

        self.a = alpha_tensor.eval(session=sess)
        return self.a

    def map_estimate_BFGS_topics(self,
                                 max_iter=1000,
                                 numTopics=2,
                                 initialize=True,
                                 batch_size=None):
        """
        Runs BFGS on the data and including the topic information
        :param max_iter: int | number of max iterations for BFGS
        :param initialize: bool | set to True if alpha values should be
        initialized
        :param batch_size: int | needed for batch update
        :return: alpha
        """
        topics = self.topics
        sess = self.sess

        max_iter = max_iter

        if initialize:
            self.batch = 0

        theta_topics = tf.reshape(
            tf.divide(tf.ones((1, numTopics)), numTopics), (numTopics, 1))

        if batch_size is None:
            Inf = genInfectedTensor(self.data, self.numNodes, self.T)
            U = genUninfectedTensor(self.data, self.numNodes, self.T)

        else:
            Inf = genInfectedTensor(self.data[self.batch:self.batch +
                                    batch_size],
                                    self.numNodes, self.T)
            U = genUninfectedTensor(self.data[self.batch:self.batch +
                                    batch_size],
                                    self.numNodes, self.T)
            self.batch += batch_size

        U_ph = tf.placeholder(tf.float32, U.shape)
        I_ph = tf.placeholder(tf.float32, Inf.shape)

        rate_intercept = tf.Variable(tf.random_uniform((self.numNodes,
                                                        self.numNodes)),
                                     dtype=tf.float32)
        rate_affinity = tf.Variable(tf.random_uniform((self.numNodes,
                                                       self.numNodes,
                                                       numTopics)),
                                    dtype=tf.float32)

        psi_1 = tf.map_fn(
            lambda x: f_psi_1_t(rate_intercept, rate_affinity,
                                x[0], x[1], self.numNodes, numTopics),
            (I_ph, topics), dtype=tf.float32)

        psi_2 = tf.map_fn(
            lambda x: f_psi_2_t(rate_intercept, rate_affinity,
                                x[0], x[1], self.numNodes, numTopics),
            (U_ph, topics), dtype=tf.float32)

        psi_3 = tf.map_fn(
            lambda x: f_psi_3_t(rate_intercept, rate_affinity,
                                x[0], x[1], self.numNodes, numTopics),
            (I_ph, topics), dtype=tf.float32)

        prior = gamma_prior_t(rate_intercept, rate_affinity,
                              theta_topics, self.numNodes, numTopics)

        log_p = -(tf.reduce_sum(psi_1) +
                  tf.reduce_sum(psi_2) +
                  tf.reduce_sum(psi_3) +
                  tf.reduce_sum(prior))

        feed_dict = {U_ph: U.eval(session=sess),
                     I_ph: Inf.eval(session=sess)}

        optimizer = tf.contrib.opt.\
            ScipyOptimizerInterface(log_p,
                                    method='L-BFGS-B',
                                    options={'maxiter': max_iter})
        if initialize:
            model = tf.global_variables_initializer()

        sess.run(model)
        optimizer.minimize(sess, feed_dict=feed_dict)

        self.a_t1 = evaluateAlpha(rate_intercept, rate_affinity,
                                  np.array([1, 0]), self.numNodes,
                                  numTopics).eval(
            session=sess).transpose().round(1)
        self.a_t2 = evaluateAlpha(rate_intercept, rate_affinity,
                                  np.array([0, 1]), self.numNodes,
                                  numTopics).eval(
            session=sess).transpose().round(1)
        return self.a_t1, self.a_t2

    def batch_update(self, batch_size=100):
        """
        if called is updating the log-probability batchwise
        :param batch_size:
        :return: alpha
        """

        a = self.map_estimate_BFGS(max_iter=1000, initialize=True,
                                   batch_size=100)
        for i in range(len(self.data) // batch_size - 1):
            a = self.map_estimate_BFGS(max_iter=1000, batch_size=100)

        return a
