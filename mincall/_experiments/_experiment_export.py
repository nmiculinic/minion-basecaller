import tensorflow as tf

with tf.Session() as sess:
    a = tf.placeholder(tf.int32, name='input_baby')
    b = tf.Variable(10)

    add = tf.add(a, b, name='sum_it_baby')

    # Run a few operations to make sure our model works
    ten_plus_two = sess.run(add, feed_dict={a: 2})
    print('10 + 2 = {}'.format(ten_plus_two))

    ten_plus_ten = sess.run(add, feed_dict={a: 10})
    print('10 + 10 = {}'.format(ten_plus_ten))

    export_path = "/tmp/export"

    model_input = tf.saved_model.utils.build_tensor_info(a)
    model_output = tf.saved_model.utils.build_tensor_info(add)

    signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"x": model_input},
        outputs={"y": model_output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_definition,
        },
    )
    builder.save()
