import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time
import re
import json

comment_re = re.compile(
    r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

def comment_replacer(match):
    start,mid,end = match.group(1,2,3)
    if mid is None:
        # single line comment
        return ''
    elif start is not None or end is not None:
        # multi line comment at start or end of a line
        return ''
    elif '\n' in mid:
        # multi line comment with line break
        return '\n'
    else:
        # multi line comment without line break
        return ' '

def remove_comments(text):
    return comment_re.sub(comment_replacer, text)

def cleanup(textIn):
  tx = []
  for line in remove_comments(textIn).split("\n"):
    lineStrip = line.strip()
    if not len(lineStrip): continue
    if any(lineStrip.startswith(s) for s in ["//", "pragma ", "import "] ): continue
    tx.append(line)

  return '\n'.join(tx)

def nextFile():
  baseurl = "https://raw.githubusercontent.com/tintinweb/smart-contract-sanctuary/master/contracts/mainnet/%s/%s.sol"
  index_file = tf.keras.utils.get_file('contracts.json', 'https://github.com/tintinweb/smart-contract-sanctuary/blob/3c4e1fe4672177eea850cda031c5b779f707b2ec/contracts/mainnet/contracts.json?raw=true')
  with open(index_file,'r') as f:

    for nr,line in enumerate(f):
      if not line.strip(): continue
      linej = json.loads(line)
      ftarget = linej["address"].replace("0x","")
      ffolder = ftarget[:2].lower()
      fname = linej["name"]

      ftotal = baseurl % (ffolder, "%s_%s"%(ftarget, fname))
      yield ftotal


class TrainingData(object):
    def __init__(self, text):
        self.text = text
        self.len = len(text)

        self.vocab = sorted(set(self.text))
        self.vocab_size = len(self.vocab)

        self.ids_from_chars = preprocessing.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)

        self.chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        self.all_ids = self.ids_from_chars(tf.strings.unicode_split(self.text, 'UTF-8'))
        self.ids_dataset = tf.data.Dataset.from_tensor_slices(self.all_ids)

    def text_from_ids(self, ids):
        return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)   

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def getSampledDataset(self, seq_length, batch_size=64, buffer_size=10000):
        self.seq_length = seq_length
        self.examples_per_epoch = len(self.text)//(self.seq_length+1)
        self.sequences = self.ids_dataset.batch(self.seq_length+1, drop_remainder=True)

        self.dataset = self.sequences.map(self.split_input_target)

        return (
            self.dataset
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
            )

    def newModel(self, embedding_dim=256, rnn_units=1024):
        self.model = MyModel(
            # Be sure the vocabulary size matches the `StringLookup` layers.
            vocab_size=len(self.ids_from_chars.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss)
        return self.model

    def train(self, dataset, epochs=15):
        # Directory where the checkpoints will be saved
        checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        self.history = self.model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
        self.predictionModel = OneStep(self.model, self.chars_from_ids, self.ids_from_chars)

    def predict(self, inputs=['contract ', 'contract ', 'abstract ', 'interface ', 'library '], targetLen=3000):
        assert(self.model and self.predictionModel)
        
        states = None
        next_char = tf.constant(inputs)
        result = [next_char]

        for _ in range(targetLen):
            next_char, states = self.predictionModel.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        return result[0].numpy().decode('utf-8')


    def save_model(self, name="one_step"):
        tf.saved_model.save(self.predictionModel, name)

    def load_model(self, name="one_step"):
        self.predictionModel = tf.saved_model.load(name)


class SolidityTrainer(object):

    @staticmethod
    def get_training_data(maxfiles=1000, maxlen=10_000_000):
        total = []
        path_to_file = ""

        for nr,dlink in enumerate(nextFile()):
            if path_to_file: os.unlink(path_to_file)
            path_to_file = tf.keras.utils.get_file("temp", dlink)
            textIn = open(path_to_file, 'rb').read().decode(encoding='utf-8')
            textIn = cleanup(textIn)
            total.append(textIn)
            if nr >= maxfiles: break
            if len("\n\n".join(total)) >= maxlen: break

        text = "\n\n".join(total)
        
        return TrainingData(text)

    

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x





class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states