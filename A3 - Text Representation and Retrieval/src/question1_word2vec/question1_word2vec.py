import nltk
nltk.download('abc')
from nltk.corpus import abc
import sys
from keras.models import Model
from keras.layers import Input, Dense, Reshape, dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from collections import Counter
import numpy as np
from keras.callbacks import Callback
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

vocab = [word.lower() for word in abc.words()]
vocab = [word for word in vocab if word.isalnum()]
freq_count= Counter(vocab).most_common()
unique_words_count = len(freq_count)

word_indices = {}
words_dictionary = {}
for word in freq_count:
  word_indices[word[0]] = len(word_indices)
  words_dictionary[len(words_dictionary)] = word[0]


data = []
for word in vocab:
  if word in words_dictionary.values(): 
    data.append(word_indices[word])
    # data.append(list(words_dictionary.keys())[list(words_dictionary.values()).index(word)])

data_size = len(data)
# print("vocab= " , vocab)

# print("count= " , freq_count)
# print("vocab_size= " , unique_words_count)
# # print("ind_dictionary= " , ind_dictionary)
# print("words_dictionary= " , words_dictionary)
# print("data= " , data)
# # print("unk_count= " , unk_count)
# print("data_size= " , data_size)

input_target, input_context = Input((1,)), Input((1,))
embedding = Embedding(unique_words_count, 300, input_length=1, name='embedding')
target, context = Reshape((300, 1))(embedding(input_target)), Reshape((300, 1))(embedding(input_context))
output = Dense(1, activation='sigmoid')( Reshape(target_shape=(1,))(dot(inputs=[target, context], axes=1)))
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='sgd')

validation_model = Model(input=[input_target, input_context], outputs=dot(inputs=[target, context], axes=1, normalize=True))
embedding_model = Model(inputs=input_target, outputs=target)

examples_to_plot = np.random.choice(1000, 10, replace=False)

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
  plt.figure(figsize=(16, 9))
  colors = cm.rainbow(np.linspace(0, 1, len(labels)))
  for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    plt.scatter(x, y, c=color, alpha=a, label=label)
    for i, word in enumerate(words):
      plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', size=8)
  plt.legend(loc=4)
  plt.title(title)
  plt.grid(True)
  if filename:
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
  plt.show()

def prep_for_plot():
  keys = []
  word_clusters = []
  embedding_clusters = []
  #iterating in valid examples
  for id in range(len(examples_to_plot)):
    curr_word = words_dictionary[examples_to_plot[id]]
    keys.append(curr_word)
    # print(curr_word, " - ")
    val_target = np.zeros((1,))
    val_context = np.zeros((1,))
    val_target[0] = examples_to_plot[id]
    similarities = np.zeros((unique_words_count,))
    for i in range(unique_words_count):
      val_context[0] = i
      similarities[i] = validation_model.predict_on_batch([val_target, val_context])
    similar_words = np.argsort(-similarities)[:20]
    embeddings = []
    words = []

    for i in range(len(similar_words)):
      # print(words_dictionary)
      # print(similar_words[i])
      words.append(words_dictionary[similar_words[i]])
      val_context[0] = similar_words[i]
      # print(words[-1], end=" ")
      embeddings.append(embedding_model.predict_on_batch(val_context).flatten())
    # print()
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

  embedding_clusters = np.array(embedding_clusters)
  n, m, k = embedding_clusters.shape
  tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
  embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
  tsne_plot_similar_words('Similar words from vocabulary', keys, embeddings_en_2d, word_clusters, 0.7)

class Visualiser(Callback):
  def on_epoch_end(self, epoch, logs=None):
    prep_for_plot()

couples, labels = skipgrams(data, vocabulary_size=data_size, window_size=3, sampling_table= make_sampling_table(data_size))
word_target, word_context = zip(*couples)
word_target, word_context = np.array(word_target, dtype="int32"), np.array(word_context, dtype="int32")

model.fit([word_target, word_context], labels, epochs=5, callbacks=[Visualiser()])

