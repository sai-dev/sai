import tensorflow as tf
import numpy as np
import csv
from matplotlib import pyplot as plt

def load_saivs():
    """Load and returns the table of match results between sai29 nets"""

    vs_csv_file = 'sai29-vs.csv'
    with open(vs_csv_file) as f:
        dataline = csv.reader(f)
        temp = next(dataline)
        n = int(temp[0])

        print("Elements:", n)
        x = np.empty((n,n))

        for i, d in enumerate(dataline):
            x[i] = np.asarray(d, dtype=np.float64)

    return x

def build_process():
    """Builds TF process"""

    x = load_saivs()
    n = np.shape(x)[1]
    xp = x[1:n,0:1]
    xm = np.transpose(x[0:1,1:n])
    xo = x[1:n,1:n]
    r = tf.Variable(tf.truncated_normal([n-1, 2], mean=0.0, stddev=1.0, dtype=tf.float64))
    rz = tf.Variable(tf.truncated_normal([1], mean=0.0, stddev=1.0, dtype=tf.float64))
#    ones = tf.ones([1, n-1], dtype = tf.float64)
#    rr = tf.matmul(r,ones)
    y = tf.multiply(xo, tf.nn.softplus(tf.subtract(tf.multiply(tf.reshape(r[:,0],[1,n-1]),
                                                               tf.reshape(r[:,1],[n-1,1])),
                                                   tf.multiply(tf.reshape(r[:,1],[1,n-1]),
                                                               tf.reshape(r[:,0],[n-1,1])))))
    zp = tf.multiply(xp, tf.nn.softplus(tf.multiply(tf.reshape(r[:,0],[n-1,1]),rz)))
    zm = tf.multiply(xm, tf.nn.softplus(-tf.multiply(tf.reshape(r[:,0],[n-1,1]),rz)))
    cost = tf.reduce_sum(y) + tf.reduce_sum(tf.add(zp, zm)) + tf.reduce_sum(tf.squared_difference(r[:,1], tf.ones([n-1,1], dtype=tf.float64))) + tf.squared_difference(rz, 1.0)

    learning_rate = 0.002
    epochs = 100000

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:

        sess.run(init)
    
        for i in list(range(epochs)):
            sess.run(optimizer)
            
                
            if i % 500 == 0:
                print(sess.run(cost))

        savez=sess.run(rz)
        save=sess.run(r)
                
    outfile='sai29-ratings-2d.csv'
    with open(outfile, "w") as file:
        file.write("0.0,")
        file.write(str(float(savez)*400.0/np.log(10)))
        file.write("\n")
        for x in list(save):
            file.write(str(float(x[0])*400.0/np.log(10)))
            file.write(",")
            file.write(str(float(x[1])*400.0/np.log(10)))
            file.write("\n")
            

if __name__ == '__main__':
    build_process()
    
