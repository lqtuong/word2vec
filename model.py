import tensorflow as tf
import numpy as np
import math, random, collections

data_index = 0


class Config():

    with open("w2w_dataset.txt", 'r') as in_file:
        dt = eval(in_file.read())
        data, vocabulary, dictionary, reversed_dict = dt['data'], dt['count'], dt['dictionary'], dt['reversed_dictionary']

    vocabulary_size = len(vocabulary)

    valid_size = 16
    valid_window = 100
    valid_example = np.random.choice(valid_window, valid_size, replace=False)

    batch_size = 128
    embedding_size = 128
    skip_window = 1 # how many words to consider left and right
    num_skip = 2 # how many times to reuse an input to generate a context
    num_steps = 100000

def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context

class Model():

    def __init__(self):
        self.config = Config()
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.config.batch_size])
        self.train_context = tf.placeholder(tf.int32, shape=[self.config.batch_size, 1])
        self.valid_dataset = tf.constant(self.config.valid_example, dtype=tf.int32)
        self.embeddings = tf.Variable(tf.random_uniform([self.config.vocabulary_size, self.config.embedding_size], -1.0, 1.0))
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

    def build(self):
        self.weights = tf.Variable(tf.truncated_normal([ self.config.vocabulary_size,  self.config.embedding_size], stddev=1.0/math.sqrt( self.config.embedding_size)))
        self.biases = tf.Variable(tf.zeros([ self.config.vocabulary_size]))
        self.hidden_out = tf.matmul(self.embed, tf.transpose(self.weights)) + self.biases
        return self.hidden_out

def train():

    model = Model()
    with tf.variable_scope("loss") as scope:
        # convert train_context to a one-hot format
        train_one_hot = tf.one_hot(model.train_context, model.config.vocabulary_size)
        print("train_one_hot ", train_one_hot.shape)
        hidden = model.build()
        print("model shape ", hidden.shape)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden, labels=train_one_hot))
        print("cross_entropy ", cross_entropy.shape)
        # Compute the cosine similarity between minibatch examples and all embeddings.
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(model.embeddings), 1, keep_dims=True))
        normalized_embedding = model.embeddings / l2_norm
        tf.summary.scalar("loss", l2_norm)
    with tf.variable_scope("learning_rate") as scope:
        batch = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(0.01, batch*model.config.batch_size, model.config.vocabulary_size, 0.99, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        tf.summary.scalar("learning_rate", learning_rate)

    valid_embeddings = tf.nn.embedding_lookup(normalized_embedding, model.valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embedding, transpose_b=True)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', session.graph)
        print("Initialized")
        average_loss = 0
        saver = tf.train.Saver()
        for step in range(model.config.num_steps):
            batch_inputs, batch_context = generate_batch(data=model.config.data, batch_size=model.config.batch_size, num_skips=model.config.num_skip, skip_window=model.config.skip_window)
            print(batch_inputs.shape)
            print(batch_context.shape)
            feed_dict = {model.train_inputs: batch_inputs,
                         model.train_context: batch_context}
            summary, _, loss_val = session.run([merge, optimizer, cross_entropy], feed_dict=feed_dict)
            writer.add_summary(summary, step)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print('Average loss step ', step, ': ', average_loss)
                average_loss = 0
                save_path = saver.save(session, "model.ckpt")

            # prints out these top_k closest words
            if step % 10000 == 0:
                # evaluates the similarity operation
                # which returns an array of cosine similarity values for each of the validation words
                sim = similarity.eval()
                for i in range(model.config.valid_size):
                    valid_word = model.config.reversed_dict[model.config.valid_example[i]]
                    top_k = 8
                    nearest = (-sum[i, :]).argsort()[1:top_k+1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = model.config.reversed_dict[nearest[k]]
                        log_str = '%s $s' % (log_str, close_word)
                    print(log_str)
        with open("w2w_output.txt", "w") as out:
            final_embeddings = normalized_embedding.eval()
            print(final_embeddings[0])
            out.write(final_embeddings)

if __name__=="__main__":
    train()