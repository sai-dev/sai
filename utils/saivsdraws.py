#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import csv
import sys
import os

GORFACTOR = 400.0/np.log(10.0)

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

def load_initial_ratings(csv_pruned_nets_file, n):
    updated_ratings = np.full((n,1), -1.0)
    
    with open(csv_pruned_nets_file, newline='') as pnetsfile:
        pnets_csv_it = csv.reader(pnetsfile)
        for i, row in enumerate(pnets_csv_it):
            updated_ratings[i,0] = float(row[-1]) / GORFACTOR

    # replace missing raings with max rating
    max_rating = updated_ratings.max()
    updated_ratings[updated_ratings == -1.0] = max_rating

    return updated_ratings

def write_ratings_file(csv_nets_file, csv_ratings_file, csv_pruned_nets_file, ratings):
    ratmap = {}
    with open(csv_pruned_nets_file, newline='') as pnetsfile:
        pnets_csv_it = csv.reader(pnetsfile)
        for i, row in enumerate(pnets_csv_it):
            hash = row[0]
            ratmap[hash] = max (0.0, ratings[i] * GORFACTOR)

    with open(csv_nets_file, newline='') as netsfile:
        with open(csv_ratings_file, "w", newline='') as ratfile:
            inrd = csv.reader(netsfile)
            outwr = csv.writer(ratfile)
            for inrow in inrd:
                hash = inrow[0]
                new_rat = ratmap.get(hash)
                if new_rat is not None:
                    # drop existing rating
                    inrow = inrow[:-1]
                    # append new rating
                    gor = float(new_rat)
                    inrow.append(f'{gor:.1f}')
                outwr.writerow(inrow)

def build_process(hook_pos: int):
    """Builds TF process"""

    global wins
    global nums
    global n
    global initial_ratings

    s_vbl = tf.Variable(initial_ratings[hook_pos+1:], dtype=tf.float32)
    # variable part of the s vector, with ratings of all nets apart from the hook_pos ones

    s0 = tf.constant(initial_ratings[0:hook_pos+1], shape=[hook_pos+1,1], dtype=tf.float32)
    # fixed part of the s vector, with rating of all nets up to hook_pos being set to previous run results

    s_scale = tf.Variable(1.0, dtype=tf.float32)

    s = tf.concat([s0, s_vbl], 0) # n*1 (column) vector with ratings
    s = tf.math.scalar_mul(s_scale, s)

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

    p_estim_per_u = 1.0 - tf.multiply(tf.math.xdivy(b_frac, h), tf.transpose(u))
    p_estim = tf.multiply(u, p_estim_per_u)
    q_estim = 1.00001 - p_estim - tf.transpose(p_estim)
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
    avg_ll = loglike / tf.reduce_sum(nums)
    excess = tf.reduce_mean(2.0 * tf.math.xlogy(p_estim, p_estim) + tf.math.xlogy(q_estim, q_estim))
    excess -= avg_ll

    learning_rate = 0.01
    epochs = 15000

#    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(-loglike)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(-loglike)
    #tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#    config = tf.ConfigProto(gpu_options=gpu_options)

#    with tf.Session(config=config) as sess:
    with tf.Session() as sess:
    
        sess.run(init)

        # print(sess.run(s))
        # print(sess.run(draws))
        print(sess.run([tf.reduce_sum(wins_loglike), tf.reduce_sum(draw_loglike)]))
    
        stable = 0
        max_ll = 1
        for i in list(range(epochs)):
            #            print(sess.run(s))
            #            print(sess.run(root))
            #            print(sess.run(tf.gradients(tf.reduce_sum(root), s)))
            sess.run(optimizer)
            #            print(sess.run(s))
            this_ll = sess.run(avg_ll)
            this_excess = sess.run(excess)
            if max_ll > 0 or this_ll > max_ll:
                max_ll = this_ll
                stable = 0
                best_s = sess.run(s)
            else:
                stable += 1
            
            if i % 20 == 0:
#                print(i, this_ll, stable)
                print(i, -this_ll, this_excess, stable)
            if stable > 1000:
                break

        return best_s


if __name__ == '__main__':
    argc = len(sys.argv)
    if not(argc == 3 or argc == 7):
        print('''Syntax:\n'''
              '''saivsdraws.py <saiXX> <hook_position>\n'''
              '''saivwdraws.py <saiXX-vs.csv> <saiXX-num.csv> <saiXX-pruned-nets.csv> '''
              '''<saiXX-nets.csv> <saiXX-ratings.csv> <hook_position>'''
              )
        exit(1)

    if argc == 3:
        vsfile = sys.argv[1] + '-vs.csv'
        numfile = sys.argv[1] + '-num.csv'
        pnetsfile = sys.argv[1] + '-pruned-nets.csv'
        netsfile = sys.argv[1] + '-nets.csv'
        ratfile = sys.argv[1] + '-ratings.csv'
        hook_pos = int(sys.argv[2])
    else:
        vsfile = sys.argv[1]
        numfile = sys.argv[2]
        pnetsfile = sys.argv[3]
        netsfile = sys.argv[4]
        ratfile = sys.argv[5]
        hook_pos = int(sys.argv[6])

    vs_e, num_e = os.path.exists(vsfile), os.path.exists(numfile)
    pnets_e, nets_e = os.path.exists(pnetsfile), os.path.exists(netsfile)

    if not vs_e:
        print(f"File {vsfile} does not exists.")
    if not num_e:
        print(f"File {numfile} does not exists.")
    if not pnets_e:
        print(f"File {pnetsfile} does not exists.")
    if not nets_e:
        print(f"File {netsfile} does not exists.")
    if not(vs_e and num_e and pnets_e and nets_e):
        exit(1)

    wins = load_table_file(vsfile)
    n = np.shape(wins)[1] # number of nets and dimension of all vectors/matrices
    nums = load_table_file(numfile)
    m = np.shape(nums)[1]
    if n != m:
        print(f"File {vsfile} has {n} networks, while file {numfile} has {m} networks.")
        exit(1)

    if np.count_nonzero(nums, axis=1).min() == 0:
        print(f"File {numfile} has rows with no matches.")
        exit(1)

    initial_ratings = load_initial_ratings(pnetsfile, n)

    final_ratings = build_process(hook_pos)

    write_ratings_file(netsfile, ratfile, pnetsfile, final_ratings)
