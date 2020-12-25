import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.datasets import mnist

np.random.seed(0)

epochs = 3
batch_size = 64
optimizer = 'adam'


def reshape(x):
    return x.reshape(x.shape[0], 28, 28, 1)


def normalize(x):
    return x / 255.0


def generate_model():
  model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
  model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
  return model


def evaluate(model, x_test, y_test):
  loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
  print("Test accuracy: ", accuracy)


def build_confusion_matrix(model, x_test, y_test, labels):
  y_predict = model.predict(x_test)
  y_predict_classes = np.argmax(y_predict, axis=1)

  conf_matrix = confusion_matrix(y_test, y_predict_classes)
  fig, ax = plt.subplots(figsize=(15, 10))
  ax = sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
  ax.set_xlabel('Predicted Label')
  ax.set_ylabel('True Label')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)


def build_image_matrix(x_test, y_test, labels):
  classes = 10
  objects = len(x_test)
  highest_scores = np.zeros(shape=(classes, classes))
  cf = np.zeros(shape=(classes, classes))
  y_predict = model.predict(x_test)

  # building confusion matrix and highest scores

  for i in range(objects):
    true_label = y_test[i]
    max_value = max(y_predict[i])
    index = np.argmax(y_predict[i])
    if max_value > highest_scores[true_label][index]:
      highest_scores[true_label][index] = max_value
      cf[true_label][index] = i

  # drawing image matrix

  for i in range(classes):
    f, ax = plt.subplots(1, classes, figsize=(28, 28))
    for j in range(classes):
      index = int(cf[i][j])
      if index == 0:
        ax[j].imshow(np.zeros(shape=(28, 28)), cmap='gray')
        continue
      sample = x_test[index]
      ax[j].imshow(sample.reshape(28, 28), cmap='gray')
      ax[j].set_title("True: {}\n Predict: {}".format(labels[i], labels[j]), fontsize=14)


(digits_x_train, digits_y_train), (digits_x_test, digits_y_test) = mnist.load_data()

digits_x_train = reshape(digits_x_train)
digits_x_test = reshape(digits_x_test)

digits_x_train = normalize(digits_x_train)
digits_x_test = normalize(digits_x_test)

model = generate_model()
model.fit(digits_x_train, digits_y_train, epochs=epochs, batch_size=batch_size)
evaluate(model, digits_x_test, digits_y_test)

digits_labels = [i for i in range(10)]
build_confusion_matrix(model, digits_x_test, digits_y_test, digits_labels)
build_image_matrix(digits_x_test, digits_y_test, digits_labels)


(fashion_x_train, fashion_y_train), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()

fashion_x_train = reshape(fashion_x_train)
fashion_x_test = reshape(fashion_x_test)

fashion_x_train = normalize(fashion_x_train)
fashion_x_test = normalize(fashion_x_test)

model = generate_model()
model.fit(fashion_x_train, fashion_y_train, epochs=epochs, batch_size=batch_size)
evaluate(model, fashion_x_test, fashion_y_test)
fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
build_confusion_matrix(model, fashion_x_test, fashion_y_test, fashion_labels)
build_image_matrix(fashion_x_test, fashion_y_test, fashion_labels)