import tensorflow as tf


"""
{'query': [[671, 2399, 5277, 5739, 6427, 0, 0, 0, 0, 0]], 
'query_len': [5],
 'doc_pos': [[5739, 6427, 3837, 1164, 6432, 0, 0, 0, 0, 0]], 'doc_pos_len': [5], 
 'doc_neg': [[671, 2399, 5277, 5739, 6427, 677, 1085, 6228, 7574, 0], 
 [671, 2399, 5277, 5739, 6427, 2110, 739, 0, 0, 0], 
 [671, 2399, 5277, 5739, 6427, 2797, 2826, 2845, 0, 0], 
 [671, 2399, 5277, 5739, 6427, 741, 0, 0, 0, 0]], 
 'doc_neg_len': [9, 7, 8, 6]}
"""


def load():
    sess = tf.Session()

    # First, load meta graph and restore weights
    saver = tf.train.import_meta_graph('model/model_1.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))

    # Second, access and create placeholders variables and create feed_dict to feed new data
    graph = tf.get_default_graph()

    # graph_op = graph.get_operations()
    # for i in graph_op:
    #     print(i)

    query_batch = graph.get_tensor_by_name('input/query_batch:0')
    doc_positive_batch = graph.get_tensor_by_name('input/doc_positive_batch:0')
    doc_negative_batch = graph.get_tensor_by_name('input/doc_negative_batch:0')
    on_training = graph.get_tensor_by_name('input/on_training:0')
    drop_out_prob = graph.get_tensor_by_name('input/drop_out_prob:0')
    query_sequence_length = graph.get_tensor_by_name('input/query_sequence_length:0')
    neg_sequence_length = graph.get_tensor_by_name('input/neg_sequence_length:0')
    pos_seq_length = graph.get_tensor_by_name('input/pos_seq_length:0')
    feed_dict = {query_batch: [[5011, 6381, 3315, 4510, 5554, 0, 0, 0, 0, 0]],
                 doc_positive_batch: [[5011, 6381, 3315, 4510, 5554, 2582, 720, 6825, 2970, 165]],
                 doc_negative_batch:[[5011, 6381, 3315, 4510, 5554, 2961, 6121, 0, 0, 0]],
                 on_training: False,
                 drop_out_prob: 1,
                 query_sequence_length: [10],
                 neg_sequence_length: [10],
                 pos_seq_length: [10]}

    # Access the op that want to run
    op_to_restore = graph.get_tensor_by_name('Cosine_Similarity/cos_sim:0')

    print(sess.run(op_to_restore, feed_dict=feed_dict))


def main():
    load()


if __name__ == '__main__':
    main()
