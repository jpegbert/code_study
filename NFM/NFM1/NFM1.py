"""
TensorFlow Implementation of <<Neural Factorization Machines for Sparse Predictive Analytics>>
"""

import tensorflow as tf
#0 1:0.05 2:0.006633 3:0.05 4:0 5:0.021594 6:0.008 7:0.15 8:0.04 9:0.362 10:0.1 11:0.2 12:0 13:0.04 15:1 555:1 1078:1 17797:1 26190:1 26341:1 28570:1 35361:1 35613:1 35984:1 48424:1 51364:1 64053:1 65964:1 66206:1 71628:1 84088:1 84119:1 86889:1 88280:1 88283:1
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(1000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def model_fn(features, labels, mode, params):
    """Build Model function f(x) for Estimator."""
    #------hyper parameters------
    field_size = params['field_size']
    feature_size = params['feature_size']
    embedding_size = params['embedding_size']
    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']
    dropout = params['dropout']
    layers = params['layers']

    #------build weights------
    Global_Bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    Feat_Wgts = tf.get_variable(name='linear', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feature------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    #------build f(x)------
    # f(x) = bias + sum(wx) + MLP(BI(embed_vec))

    # FM部分
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Feat_Wgts, feat_ids) # None * F * 1
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)  # None * 1

    with tf.variable_scope("BiInter-part"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) # None * F * k
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1]) # None * F * 1
        embeddings = tf.multiply(embeddings, feat_vals) # vi * xi
        sum_square_emb = tf.square(tf.reduce_sum(embeddings, 1))
        square_sum_emb = tf.reduce_sum(tf.square(embeddings), 1)
        deep_inputs = 0.5 * tf.subtract(sum_square_emb, square_sum_emb) # None * k

    with tf.variable_scope("Deep-part"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        # BI的输出需要进行Batch Normalization
        deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn="bn_after_bi")

        # BI的输出进行Dropout
        if mode == tf.estimator.ModeKeys.TRAIN:
            deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[-1]) # dropout at bilinear interaction layer

        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i], weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope="mlp%d" % i)
            # 注意是先进行Batch Norm，再进行Dropout
            # Batch Normalization
            deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn="bn%d" % i)
            # Dropout
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        # Output
        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope="deep_out")
        y_d = tf.reshape(y_deep, shape=[-1])

    with tf.variable_scope("NFM-out"):
        y_bias = Global_Bias * tf.ones_like(y_d, dtype=tf.float32)
        y = y_bias + y_linear + y_d
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}

    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    #------build loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) + l2_reg * tf.nn.l2_loss(Feat_Wgts) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    #------build optimizer------
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())


    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


#---------------------------------------------------------------------------------------------------------------
#------------------------------------------ Main Function ------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# 日志级别
tf.logging.set_verbosity(tf.logging.INFO)

print("构建分类器......")
model_params = {
    'field_size': 13 + 26,
    'feature_size': 117581,
    'embedding_size': 64,
    'l2_reg': 0.0005,
    'learning_rate': 0.001,
    'layers':[400,400,400],
    'dropout':[0.5, 0.5, 0.5] # 最后一个是BI输出的keep_prob
}

log_steps = 1000
config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': 10}),
    log_step_count_steps=log_steps, save_summary_steps=log_steps)
classifier = tf.estimator.Estimator(model_fn=model_fn,model_dir='./model_save', params=model_params, config=config)  # Path to where checkpoints etc are stored

train_file = '../EveryTestInOne/criteo/tr.libsvm'
test_file = '../EveryTestInOne/criteo/te.libsvm'
val_file = '../EveryTestInOne/criteo/va.libsvm'


print("训练......")
# 500 epochs = 500 * 120 records [60000] = (500 * 120) / 32 batches = 1875 batches
# 4 epochs = 4 * 30 records = (4 * 30) / 32 batches = 3.75 batches
classifier.train(input_fn=lambda: input_fn(train_file, 256, 1, True))

print("评估......")
evaluate_result = classifier.evaluate(input_fn=lambda: input_fn(val_file, 256, 1, False))
for key in evaluate_result:
    tf.logging.info("{}, was: {}".format(key, evaluate_result[key]))

print("预测......")
predict_results = classifier.predict(input_fn=lambda: input_fn(test_file, 256, 1, False))
tf.logging.info("Prediction on test file")
for prediction in predict_results:
    tf.logging.info("{}".format(prediction["prob"]))
    break


# eval on Test
evaluate_result_test = classifier.evaluate(input_fn = lambda:input_fn(test_file, 256, 1, False))
for key in evaluate_result_test:
    tf.logging.info("{0}, was: {1}".format(key, evaluate_result_test[key]))
