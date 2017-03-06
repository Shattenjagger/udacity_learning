"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
import tensorflow as tf

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x')


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name='keep_prob')


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    s = [conv_ksize[0], conv_ksize[1], x_tensor.shape[3].value, conv_num_outputs]
    print(s)
    weights = tf.Variable(tf.random_normal(shape=s))
    b = tf.Variable(tf.random_normal([conv_num_outputs]))
    out = tf.nn.conv2d(x_tensor, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    out = tf.nn.relu(out)
    out = tf.nn.max_pool(out, ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                         strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    return out


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    old_shape = x_tensor.get_shape().as_list()
    new_shape = [-1, old_shape[1] * old_shape[2] * old_shape[3]]
    return tf.reshape(x_tensor, new_shape)


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    w = tf.Variable(tf.random_normal([x_tensor.shape[1].value, num_outputs]))
    b = tf.Variable(tf.random_normal([num_outputs]))
    out = tf.add(tf.matmul(x_tensor, w), b)
    out = tf.nn.relu(out)
    return out


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    w = tf.Variable(tf.random_normal([x_tensor.shape[1].value, num_outputs]))
    b = tf.Variable(tf.random_normal([num_outputs]))
    out = tf.add(tf.matmul(x_tensor, w), b)
    return out


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    out = conv2d_maxpool(x, 32, [3, 3], [1, 1], [2, 2], [1, 1])
    # out = tf.layers.dropouout, keep_prob)
    out = conv2d_maxpool(out, 64, [3, 3], [1, 1], [2, 2], [1, 1])
    # out = tf.layers.dropout(out, keep_prob)
    out = conv2d_maxpool(out, 128, [3, 3], [1, 1], [2, 2], [1, 1])
    out = tf.layers.dropout(out, keep_prob)

    out = flatten(out)

    out = fully_conn(out, 512)
    out = tf.layers.dropout(out, keep_prob)
    # out = fully_conn(out, 64)
    # out = tf.layers.dropout(out, keep_prob)
    # out = fully_conn(out, 32)
    # out = tf.layers.dropout(out, keep_prob)

    out = output(out, 10)
    return out

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={y: label_batch, x: feature_batch, keep_prob: keep_probability})


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    global valid_features, valid_labels
    test_valid_size = 512
    loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.})
    valid_acc = session.run(accuracy, feed_dict={x: valid_features[:test_valid_size], y: valid_labels[:test_valid_size], keep_prob: 1.})
    print("Loss: {:>10.4f} Validation Accuracy: {:.6f}".format(loss, valid_acc))


epochs = 120
batch_size = 256
keep_probability = 0.5

print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
