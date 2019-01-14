import tensorflow as tf
import numpy as np
import csv
#from matplotlib import pyplot as plt

def load_table_file(csv_table_file):
    """Load and returns the table of match results between sai29 nets"""

    with open(csv_table_file) as f:
        dataline = csv.reader(f)
        temp = next(dataline)
        n = int(temp[0])

        print("Elements:", n)
        x = np.empty((n,n), dtype=np.float32)

        for i, data in enumerate(dataline):
            x[i] = np.asarray(data, dtype=np.float32)

    return x

def build_process():
    """Builds TF process"""


    wins = load_table_file('sai30-vs.csv')
    nums = load_table_file('sai30-num.csv')


    n = np.shape(wins)[1] # number of nets and dimension of all vectors/matrices

    s_vbl = tf.Variable(tf.truncated_normal([n-1, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
    # variable part of s vector, with ratings of nets different from the first one

    s0 = tf.constant(0.0, shape=[1,1], dtype=tf.float32)
    # the rating of the first network is set to 0 wlog

    s = tf.concat([s0, s_vbl], 0) # n*1 (column) vector with ratings
    
    ones = tf.constant(1.0, shape=[1, n], dtype = tf.float32) # row vector with ones
    
    s_mtx = tf.matmul(s,ones) # square matrix with column vectors s

    d = tf.subtract(s_mtx, tf.transpose(s_mtx)) # anti-symmetric matrix with differences
    # d_ij = s_i - s_j

    u = tf.sigmoid(d) # u = 1 / (1 + exp(-d)) componentwise

    rho = tf.math.xdivy(wins, nums)

    beta = tf.math.xdivy(tf.subtract(tf.subtract(nums, wins), tf.transpose(wins)), nums)

    ru = tf.multiply(rho, u)

    tau = tf.subtract(tf.subtract(tf.ones([n,n], dtype=tf.float32), ru), tf.transpose(ru))

    fbuu = 4 * tf.multiply(beta, tf.multiply(u, tf.transpose(u)))

    q = tf.subtract(tau, tf.sqrt(tf.subtract(tf.square(tau),  fbuu)))

    win_loglike = tf.math.xlogy(wins, tf.subtract(u, 0.5 * q))
    draw_loglike = tf.math.xlogy(tf.multiply(nums, beta), q)
    loglike = tf.reduce_sum(win_loglike) + tf.reduce_sum(draw_loglike)

    learning_rate = 0.002
    epochs = 20000

#    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(-loglike)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(-loglike)
    #tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#    config = tf.ConfigProto(gpu_options=gpu_options)

#    with tf.Session(config=config) as sess:
    with tf.Session() as sess:

    
        sess.run(init)
        print(sess.run(tf.reduce_sum(win_loglike)))
        print(sess.run(tf.reduce_sum(draw_loglike)))
    
        for i in list(range(epochs)):
            sess.run(optimizer)
            
            if i % 100 == 0:
                print(sess.run(loglike))

        save=sess.run(s)
                
    outfile='sai30-ratings'
    with open(outfile, "w") as file:
        file.write("0.0\n")
        for x in list(save):
            file.write(str(float(x)*400.0/np.log(10)))
            file.write("\n")
            

if __name__ == '__main__':
    build_process()
    
