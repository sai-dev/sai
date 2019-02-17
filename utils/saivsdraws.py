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
    # variable part of s vector, with ratings of all nets apart from the first one

    s0 = tf.constant(0.0, shape=[1,1], dtype=tf.float32)
    # the rating of the first network is set to 0 wlog

    s = tf.concat([s0, s_vbl], 0) # n*1 (column) vector with ratings
    
    ones = tf.constant(1.0, shape=[1, n], dtype = tf.float32) # row vector with ones
    
    s_mtx = tf.matmul(s,ones) # square matrix with column vectors s

    d = tf.subtract(s_mtx, tf.transpose(s_mtx)) # anti-symmetric matrix with differences
    # d_ij = s_i - s_j

    u = tf.sigmoid(d) # u = 1 / (1 + exp(-d)) componentwise
    minus_log_u = tf.math.softplus(-d)

    #    rho = tf.math.xdivy(wins, nums)

    w_frac = tf.math.xdivy(wins, nums)
    b_frac = tf.subtract(tf.subtract(1.0, w_frac), tf.transpose(w_frac))
    draws = tf.subtract(tf.subtract(nums, wins), tf.transpose(wins))   #  b

    b_plus_c_u = 0.5 * tf.multiply(u, tf.subtract(1.0, w_frac))  # (b+c)u / 2
    ##    b_plus_c_u = 0.5 * tf.multiply(u, tf.subtract(nums, wins))  # (b+c)u / 2
    a_plus_b_v = tf.transpose(b_plus_c_u)                       # (a+b)v / 2

    au = tf.multiply(u, w_frac)                            # au
    ##    au = tf.multiply(u, wins)                            # au
    cv = tf.transpose(au)                                # cv
    root = tf.sqrt(0.00001 + tf.eye(n) + tf.add(tf.square(tf.subtract(b_plus_c_u, a_plus_b_v)), tf.multiply(au, cv)))

    #    tau_num = tf.add(b_plus_c_u, a_plus_b_v)             # (av+b+cu) / 2
    h = tf.add(tf.add(b_plus_c_u, a_plus_b_v), root)
    b_softplus = tf.multiply(draws, minus_log_u)

    draw_loglike = tf.math.xlogy(draws, b_frac) + draws * np.log(2) - b_softplus - tf.transpose(b_softplus) - tf.math.xlogy(draws, h)
    wins_loglike = tf.math.xlogy(wins, 1.0 - tf.multiply(tf.math.xdivy(b_frac, h), tf.transpose(u))) - tf.multiply(wins, minus_log_u)

    #    beta = tf.math.xdivy(draws, nums)
    #    ru = tf.multiply(rho, u)
    #    tau = tf.subtract(tf.subtract(tf.ones([n,n], dtype=tf.float32), ru), tf.transpose(ru))
    #    fbuu = 4 * tf.multiply(beta, tf.multiply(u, tf.transpose(u)))
    #    log_fuu = np.log(4) - minus_log_u + tf.transpose(minus_log_u)
    #    root_sum = tf.add(tau, tf.sqrt(tf.subtract(tf.square(tau),  fbuu)))

    #    q = tf.subtract(tau, tf.sqrt(tf.subtract(tf.square(tau),  fbuu)))
    #    q = tf.divide(fbuu, tf.add(tau, tf.sqrt(tf.subtract(tf.square(tau),  fbuu))))
    #    q = tf.multiply(beta, tf.divide(tf.exp(log_fuu), root_sum))
    #    beta_log_q = tf.add(tf.math.xlogy(beta, beta), tf.multiply(beta, log_fuu - tf.log(root_sum))

    #    win_loglike = tf.math.xlogy(wins, tf.subtract(u, 0.5 * q))
    #    draw_loglike = tf.math.xlogy(tf.multiply(nums, beta), q)

    loglike = 2.0 * tf.reduce_sum(wins_loglike) + tf.reduce_sum(draw_loglike)

    learning_rate = 0.004
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
        print(sess.run([tf.reduce_sum(wins_loglike), tf.reduce_sum(draw_loglike)]))
    
        for i in list(range(epochs)):
            #            print(sess.run(s))
            #            print(sess.run(root))
            #            print(sess.run(tf.gradients(tf.reduce_sum(root), s)))
            sess.run(optimizer)
            #            print(sess.run(s))
            
            if i % 100 == 0:
                print(sess.run(loglike))

        save=sess.run(s)
                
    outfile='sai30-ratings'
    with open(outfile, "w") as file:
        for x in list(save):
            file.write(str(float(x)*400.0/np.log(10)))
            file.write("\n")
            

if __name__ == '__main__':
    build_process()
    
