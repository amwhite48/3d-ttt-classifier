import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import data.shape_dataset.shape_dataset
import matplotlib.pyplot as plt


def convert_dataset(item):
    """Puts the shape dataset in the format Keras expects, (features, labels)."""
    voxels = item['voxels']
    l = item['label']
    label =tf.one_hot(l, 3)
    return voxels, label

train, test = tfds.load('shape_dataset', split=['train[:75%]', 'train[75%:]'])
train = train.map(convert_dataset).batch(100).cache()
test = test.map(convert_dataset).batch(100).cache()

leaky_relu = tf.keras.layers.LeakyReLU(0.2)
relu = tf.keras.layers.Activation(activation='relu')

def get_model(name="model"):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(16, 16, 16)),
        tf.keras.layers.Reshape((16, 16, 16, 1)),
        tf.keras.layers.Conv3D(256, (4,4,4), 1, kernel_initializer='glorot_normal', padding='same'),
        leaky_relu,
        tf.keras.layers.Conv3D(128, (4,4,4), 2, kernel_initializer='glorot_normal', padding='same'),
        leaky_relu,
        tf.keras.layers.Conv3D(64, (4,4,4), 1, kernel_initializer='glorot_normal', padding='valid'),
        leaky_relu,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ], name=name)

class Classifier_Model(tf.keras.Model):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    
    def call(self, inputs, **kwargs):
        return self.model(inputs)

m = Classifier_Model(    
    model = get_model(), 
    name="3d_shape_classifier"
)

m.build(input_shape=(200, 16, 16, 16, 1))

m.compile(
    optimizer   = tf.keras.optimizers.Adam(learning_rate=0.005),
    loss        = tf.keras.losses.CategoricalCrossentropy(),
    metrics     = [tf.keras.metrics.CategoricalAccuracy()]
)

m.model.summary()


history = m.fit(
    train,
    validation_data = test, 
    epochs     = 1, 
    batch_size = 100
)

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('val_accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

