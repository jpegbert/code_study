import tensorflow as tf
import math


def build_model(user_indices, item_indices, rank, ratings, user_cnt, item_cnt, lr, lamb, mu, init_value):
    W_user = tf.Variable(tf.truncated_normal([user_cnt, rank], stddev=init_value /math.sqrt(float(rank)), mean=0),
                         name='user_embedding', dtype=tf.float32)
    W_item = tf.Variable(tf.truncated_normal([item_cnt, rank], stddev=init_value /math.sqrt(float(rank)), mean=0),
                         name='item_embedding', dtype=tf.float32)

    W_user_bias = tf.concat([W_user, tf.ones((user_cnt, 1), dtype=tf.float32)], 1, name='user_embedding_bias')
    W_item_bias = tf.concat([tf.ones((item_cnt, 1), dtype=tf.float32), W_item], 1, name='item_embedding_bias')

    user_feature = tf.nn.embedding_lookup(W_user_bias, user_indices, name='user_feature')
    item_feature = tf.nn.embedding_lookup(W_item_bias, item_indices, name='item_feature')

    preds = tf.add(tf.reduce_sum(tf.multiply(user_feature, item_feature), 1), mu)

    square_error = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, ratings)))
    loss = square_error + lamb * (tf.reduce_mean(tf.nn.l2_loss(W_user)) + tf.reduce_mean(tf.nn.l2_loss(W_item)))

    tf.summary.scalar('square_error', square_error)
    tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()
    # tf.global_variables_initializer()
    # tf.train.AdadeltaOptimizer(learning_rate=lr).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    return train_step, square_error, loss, merged_summary

