import tensorflow as tf
import edward as ed
from edward.models import Normal, Gamma, MultivariateNormalTriL
from edward.models import PointMass


## Initialize the values 

N = 5
c = 2

# Cascades should be of size c x n; Initialize the variables
# TO DO: make A only NxN not c x NxN !!!!
C = tf.placeholder(tf.float32, [c,N])
A = Gamma(concentration=tf.zeros((c,N,N)), rate=tf.ones((c,N,N)))

I_inf = tf.expand_dims(tf.where(C>0, x=tf.ones((c,N)), y=tf.zeros((c,N))),1)
I_uninf = 1 - tf.expand_dims(tf.where(C>0, x=tf.ones((c,N)), y=tf.zeros((c,N))),1)
t_max = tf.expand_dims(tf.reduce_max(C,1),1)

#calculate dist1
t_dist1 = tf.expand_dims(tf.subtract(t_max,C),1)

#calculate dist2
k = tf.concat([-tf.ones((c,N,1)),tf.expand_dims(C,-1)], 2)
k_T = tf.transpose(tf.concat([tf.expand_dims(C,-1),tf.ones((c,N,1))], 2),perm=[0,2,1])
t_dist2 = tf.matmul(k,k_T)

# calculate psi_1 

psi_1 = tf.multiply(tf.multiply(A,I_inf), tf.transpose(I_uninf, perm=[0,2,1]))
psi_1 = tf.multiply(psi_1,t_dist1)
psi_1 = tf.reduce_sum(-psi_1)

# calculate psi_2

psi_2 = tf.multiply(tf.where(t_dist2 > 0, A, tf.zeros((c,N,N))),tf.transpose(I_inf, perm=[0,2,1]))
psi_2 = tf.multiply(psi_2,t_dist2)
psi_2 = tf.reduce_sum(-psi_2)

# calculate psi_3

psi_3 = tf.multiply(tf.where(t_dist2 > 0, A, tf.zeros((c,N,N))), I_inf)
psi_3 = tf.transpose(tf.log(tf.reduce_sum(psi_3,axis=2,keepdims=True)), perm=[0,2,1])
psi_3 = tf.where(tf.logical_not(tf.is_inf(psi_3)), psi_3 , tf.zeros((c,1,N)))
psi_3 = tf.reduce_sum(-psi_3)

# log(p)

log_p = psi_1 + psi_2
c_i = tf.exp(log_p)

I_uninf = tf.ones((c,N)) - I_inf

# define log(p(c,A)) for a single cascade
data={C: [[0,0,1,2,3],[0,1,0,2,3]]}
sess = tf.Session()
sess.run(log_p, feed_dict=data)
