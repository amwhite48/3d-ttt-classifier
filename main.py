import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import data.shape_dataset.shape_dataset



train, test = tfds.load('shape_dataset', split=['train[:75%]', 'train[75%:]'])
train = train.cache()
train = train.batch(200)
test = test.batch(200)
test = test.cache()


# for thing in train_np:
#     print(thing)
# # train_data = np.asarray(list(map(lambda x: x['voxels'], ds)))




leaky_relu = tf.keras.layers.LeakyReLU(0.2)
relu = tf.keras.layers.Activation(activation='relu')

def get_model(name="model"):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(16, 16, 16, 1)),
        tf.keras.layers.Conv3D(512, (4,4,4), 1, kernel_initializer='glorot_normal', padding='same'),
        leaky_relu,
        tf.keras.layers.Conv3D(256, (4,4,4), 2, kernel_initializer='glorot_normal', padding='same'),
        leaky_relu,
        tf.keras.layers.Conv3D(128, (4,4,4), 1, kernel_initializer='glorot_normal', padding='valid'),
        leaky_relu,
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ], name=name)

class Classifier_Model(tf.keras.Model):

    def __init__(self, model, **kwargs):
        '''
        self.gen_model = generator model;           z_like -> x_like
        self.dis_model = discriminator model;       x_like -> probability
        self.z_sampler = sampling strategy for z;   z_dims -> z
        self.z_dims    = dimensionality of generator input
        '''
        super().__init__(**kwargs)
        self.model = model

    
    def call(self, inputs, **kwargs):
        expanded_dims = tf.expand_dims(inputs, -1)
        return self.model(expanded_dims)

    # def build(self, **kwargs):
    #     super().build(**kwargs)
    
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)


    # def test_step(self, data): 
    #     x_real = data
    #     batch_size = tf.shape(x_real)[0]

    #     x_fake = self.generate(self.sample_z(batch_size))
    #     d_fake = self.discriminate(x_fake)
    #     d_real = self.discriminate(tf.expand_dims(x_real, -1))

    #     all_funcs = {**self.loss_funcs, **self.acc_funcs}
    #     return { key : fun(d_fake, d_real) for key, fun in all_funcs.items() }

    # def train_step(self, data):
    #     x_real = data
    #     batch_size = x_real.shape[0]

    #     sample = self.sample_z(batch_size)
          
    #     loss_fn   = self.loss_funcs['d_loss']
    #     optimizer = self.optimizers['d_opt']
    #     for i in range(self.d_steps):
    #       with tf.GradientTape() as tape:
    #         x_fake = self.generate(sample)
    #         d_fake = self.discriminate(x_fake)
    #         d_real = self.discriminate(tf.expand_dims(x_real, -1))
    #         loss = loss_fn(d_fake, d_real)
    #       gradients = tape.gradient(loss, self.dis_model.trainable_variables)
    #       optimizer.apply_gradients(zip(gradients, self.dis_model.trainable_variables))

    #     loss_fn   = self.loss_funcs['g_loss']
    #     optimizer = self.optimizers['g_opt'] 

    #     for i in range(self.g_steps):
    #       with tf.GradientTape() as tape:
    #         x_fake = self.generate(sample)
    #         d_fake = self.discriminate(x_fake)
    #         d_real = self.discriminate(tf.expand_dims(x_real, -1))
    #         loss = loss_fn(d_fake, d_real)
    #       gradients = tape.gradient(loss, self.trainable_variables)
    #       optimizer.apply_gradients(zip(gradients, self.gen_model.trainable_variables))

    #     all_funcs = {**self.loss_funcs, **self.acc_funcs}
    #     return { key : fun(d_fake, d_real) for key, fun in all_funcs.items() }


m = Classifier_Model(    
    model = get_model(), 
    name="3d_shape_classifier"
)

class EpochVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_inputs, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.sample_inputs = sample_inputs
        self.imgs = [] 

    def on_epoch_end(self, epoch, logs=None):
        x_real, z_samp = self.sample_inputs
        x_fake = self.model.gen_model(z_samp)


m.build(input_shape=(200, 16, 16, 16, 1))

m.compile(
    optimizer   = tf.keras.optimizers.Adam(learning_rate=0.005),
    loss        = tf.keras.losses.CategoricalCrossentropy(),
    metrics     = [tf.keras.metrics.CategoricalAccuracy()]
)


m.fit(
    train,
    validation_data = test, 
    epochs     = 40, ## Feel free to bump this up to 20 when your architecture is done
    batch_size = 200
    # callbacks  = [viz_callback]
)

m.model.summary()

