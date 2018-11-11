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
    r = tf.Variable(tf.truncated_normal([n-1, 1], mean=0.0, stddev=1.0, dtype=tf.float64))
    ones = tf.ones([1, n-1], dtype = tf.float64)
    rr = tf.matmul(r,ones)
    y = tf.multiply(xo, tf.nn.softplus(tf.subtract(tf.transpose(rr), rr)))
    zp = tf.multiply(xp, tf.nn.softplus(-r))
    zm = tf.multiply(xm, tf.nn.softplus(r))
    cost = tf.add(tf.reduce_sum(y), tf.reduce_sum(tf.add(zp, zm)))

    learning_rate = 0.001
    epochs = 50000

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    points = [[], []]

    with tf.Session() as sess:

        sess.run(init)
    
        for i in list(range(epochs)):
            sess.run(optimizer)
            
            if i % 10 == 0.:
                points[0].append(i+1)
                points[1].append(sess.run(cost))
                
            if i % 100 == 0:
                print(sess.run(r[:3]))
                print(sess.run(r[-3:]))
                #                plt.plot(range(308),sess.run(r), 'r--')
                #                plt.show()
                #                print(sess.run(cost))

        save=sess.run(r)
                
    plt.plot(points[0], points[1], 'r--')
#    plt.axis([0, epochs])
    plt.show()

    outfile='sai29-ratings'
    with open(outfile, "w") as file:
        file.write("0.0\n")
        for x in list(save):
            file.write(str(float(x)))
            file.write("\n")
            

if __name__ == '__main__':
    build_process()
    
