import os

os.environ["tf.keras_BACKEND"] = "tensorflow"

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import layers
#from tf.keras import ops
import numpy as np
import tensorflow_datasets as tfds
from keras.models import Model
import sys
from keras import ops

@keras.saving.register_keras_serializable()
class AdamW(tf.keras.optimizers.Adam):
    def __init__(self, learning_rate=0.001, weight_decay=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
        # Initialize Adam optimizer with the same parameters
        super(AdamW, self).__init__(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs)
        self.weight_decay = weight_decay  # Store weight decay value

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        # Apply the weight decay to the weights before applying the gradients
        grads_and_vars = [
            (grad + self.weight_decay * var, var) if grad is not None else (None, var)
            for grad, var in grads_and_vars
        ]
        # Call the super class's method to apply the modified gradients
        return super(AdamW, self).apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({'weight_decay': self.weight_decay})
        return config



# tf.config.set_soft_device_placement(True)


# data
dataset_name = "places365_small"
dataset_repetitions = 7
num_epochs = 75  # train for at least 50 epochs for good results
num_epochs_flowers = 1
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 3
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 100, 128]
block_depth = 2

# optimization
batch_size = 64
ema = 0.999
transfer_learning_rate = 1e-3
learning_rate = 1e-3
weight_decay = 1e-4

attention_type = "self"

def AttentionBlock(width):
    def apply(x):
        query = layers.Conv2D(width, kernel_size=1)(x)
        key = layers.Conv2D(width, kernel_size=1)(x)
        value = layers.Conv2D(width, kernel_size=1)(x)
        
        # Compute attention
        attention_scores = tf.keras.layers.Attention()([query, key])
        attention_output = layers.Multiply()([attention_scores, value])
        
        # Residual connection
        return layers.Add()([x, attention_output])
    
    return apply

def SelfAttentionBlock(channels):
    """Self-Attention Block to enhance feature representation."""
    def apply(x):
        # Compute query, key, and value
        query = layers.Conv2D(channels // 8, kernel_size=1)(x)
        key = layers.Conv2D(channels // 8, kernel_size=1)(x)
        value = layers.Conv2D(channels, kernel_size=1)(x)

        # Calculate attention scores
        attention_scores = tf.nn.softmax(tf.matmul(
            tf.reshape(query, [tf.shape(x)[0], -1, channels // 8]),  # Flatten spatial dimensions
            tf.reshape(key, [tf.shape(x)[0], -1, channels // 8]), transpose_b=True
        ))

        # Compute attention-weighted output
        attention_output = tf.matmul(attention_scores, 
                                      tf.reshape(value, [tf.shape(x)[0], -1, channels]))
        attention_output = tf.reshape(attention_output, tf.shape(x))
        return layers.Add()([x, attention_output])  # Residual connection

    return apply


def SEBlock(channels, reduction=16):
    """Squeeze-and-Excitation Block for channel attention."""
    def apply(x):
        # Squeeze
        squeeze = layers.GlobalAveragePooling2D()(x)
        squeeze = layers.Dense(channels // reduction, activation="relu")(squeeze)
        squeeze = layers.Dense(channels, activation="sigmoid")(squeeze)

        # Scale
        scale = layers.Reshape((1, 1, channels))(squeeze)
        return layers.Multiply()([x, scale])

    return apply

class NullWriter:
    def write(self, _): pass

def suppress_output():
    sys.stdout = NullWriter()

def restore_output():
    sys.stdout = sys.__stdout__


def TransformerBlock(channels, num_heads=4, ff_dim=256):
    """Transformer Block with Multi-Head Attention and Feedforward layers."""
    def apply(x):
        # Layer Normalization
        x_norm = layers.LayerNormalization()(x)

        # Multi-Head Self-Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels)(x_norm, x_norm)
        attention_output = layers.Add()([x, attention_output])  # Residual connection

        # Feedforward
        ff_output = layers.Dense(ff_dim, activation="relu")(attention_output)
        ff_output = layers.Dense(channels, name="transformer_dense")(ff_output)
        return layers.Add()([attention_output, ff_output])  # Residual connection

    return apply


def preprocess_image(image_size = 128):
    def  preprocess_function(data):
        # center crop image
        height = tf.shape(data["image"])[0]
        width = tf.shape(data["image"])[1]
        crop_size = tf.minimum(height, width)
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )

        # resize and clip
        # for image downsampling it is important to turn on antialiasing
        image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return preprocess_function


def prepare_dataset(split, image_size = 128, add_to_ds = None):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    data = tfds.load(dataset_name, split=split, shuffle_files=False)
    data = tf.data.Dataset.from_tensor_slices(list(data.map(preprocess_image(image_size), num_parallel_calls=tf.data.AUTOTUNE)))

    if add_to_ds!=None:
        data = data.concatenate(add_to_ds)
    data = data.cache().repeat(dataset_repetitions).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return data


@keras.saving.register_keras_serializable()
class KID(tf.keras.metrics.Metric):
    def __init__(self, name, image_size, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = tf.keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype="float32")
        return (
            features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0
        ) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = real_features.shape[0]
        batch_size_f = tf.cast(batch_size, dtype="float32") 
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


@keras.saving.register_keras_serializable()
def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace( 
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = tf.cast(2.0 * math.pi * frequencies, "float32")
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width, activation="relu", normalization="batch"):
    def apply(x):
        input_width = x.shape[-1]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)

        if normalization == "batch":
            x = layers.BatchNormalization()(x)
        elif normalization == "layer":
            x = layers.LayerNormalization()(x)

        x = layers.Activation(activation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, pool_type="average", dropout_rate=0.1):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)

        if pool_type == "average":
            x = layers.AveragePooling2D(pool_size=2)(x)
        elif pool_type == "max":
            x = layers.MaxPooling2D(pool_size=2)(x)

        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

        return x

    return apply


def fix_shape_mismatch(x, skip):
    """Adjust the shapes of the feature maps by either cropping or padding."""
    if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
        if x.shape[1] > skip.shape[1] or x.shape[2] > skip.shape[2]:
            # Crop x to match the skip dimensions
            dim = (float(x.shape[1]) - float(skip.shape[1])) / 2.0
            dim1 = math.ceil(dim)
            dim2 = math.floor(dim)
            cropping = (dim1, dim2)
            x = layers.Cropping2D(((cropping, cropping)))(x)
            return x
        else:
            # Pad x to match the skip dimensions
            dim = (float(skip.shape[1]) - float(x.shape[1])) / 2.0
            dim1 = math.ceil(dim)
            dim2 = math.floor(dim)
            padding = (dim1, dim2)
            x = layers.ZeroPadding2D(((padding, padding)))(x)
            return x
    return x

def UpBlock(width, block_depth, dropout_rate=0.1):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            skip = skips.pop()
            x = fix_shape_mismatch(x, skip)
            x = layers.Concatenate()([x, skip])
            x = ResidualBlock(width)(x)
            if dropout_rate:
                x = layers.Dropout(dropout_rate)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth, attention_type="transformer", pool_type="average", dropout_rate=0.1):

    noisy_images = tf.keras.Input(shape=(image_size, image_size, 3))
    noise_variances = tf.keras.Input(shape=(1, 1, 1))

    # Sinusoidal embedding
    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    data_augmentation = keras.Sequential(
                    [
                        layers.Rescaling(1.0/255.0),
                        layers.RandomFlip("horizontal"),
                        layers.GaussianNoise(0.0001),
                        tf.keras.layers.RandomBrightness(0.0001),
                        layers.Rescaling(255.0),
                    ]
                )
    noisy_images = data_augmentation(noisy_images)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    # Downsampling path
    for width in widths[:-1]:
        x = DownBlock(width, block_depth, pool_type=pool_type, dropout_rate=dropout_rate)([x, skips])

    # Bottleneck with attention
    for i in range(block_depth):
        if i == block_depth // 2:  # Add attention at the midpoint
            if attention_type == "self":
                x = SelfAttentionBlock(widths[-1])(x)
            elif attention_type == "transformer":
                print("transformer block")
                x = TransformerBlock(widths[-1])(x)
        x = ResidualBlock(widths[-1])(x)

    # Upsampling path
    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth, dropout_rate=dropout_rate)([x, skips])

    x = layers.Conv2D(3, kernel_size=1)(x)

    return tf.keras.Model([noisy_images, noise_variances], x, name="attention_residual_unet")

@keras.saving.register_keras_serializable()
class DiffusionStandardModel(tf.keras.Model):
    def __init__(self, image_size, widths=widths, block_depth=block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = tf.keras.models.clone_model(self.network)
        self.image_size = image_size

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid", image_size=self.image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.cast(tf.math.acos(max_signal_rate), "float32")
        end_angle = tf.cast(tf.math.acos(min_signal_rate), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(
            shape=(num_images, self.image_size, self.image_size, 3)
        )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        def close_event():
            plt.close()
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
                #plt.imsave(f"./Sample{index}", generated_images[index])
        plt.tight_layout()
        timer = fig.canvas.new_timer(interval = 1000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        timer.start()
        plt.savefig(self.img_name)
        plt.show()
        plt.close()

    def plot_dataset(self, ds, num_rows=3, num_cols=6):
        def close_event():
            plt.close()
        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(ds[index])
                plt.axis("off")
        plt.tight_layout()
        timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
        timer.add_callback(close_event)
        timer.start()
        plt.show()
        plt.close()


    def learn_on_custom_dataset(self, train_dataset, val_dataset, n_images = 100, plot_imgs = True, aug=False, save=True, get_pretrained=False, lamp=False, culture=0, category=0, imb=0, percent=0.0): 
        # below tensorflow 2.9:
        # pip install tensorflow_addons
        # import tensorflow_addons as tfa
        # optimizer=tfa.optimizers.AdamW
        
        if aug:
            suppress_output()
            data_augmentation = keras.Sequential(
                    [
                        layers.Rescaling(1.0/255.0),
                        layers.RandomFlip("horizontal"),
                        layers.RandomRotation(0.2),
                        layers.GaussianNoise(0.1),
                        tf.keras.layers.RandomBrightness(0.1),
                        layers.RandomZoom(0.02, 0.02),
                        layers.Rescaling(255.0),
                    ]
                )
                
            for i in range(1):
                aug_images = data_augmentation(train_dataset)
                for img in aug_images:
                    img = tf.clip_by_value(img, 0, 255)
                    img = tf.cast(img, "uint8")
                    train_dataset.append(img)

            restore_output()
            del data_augmentation

            
        if not get_pretrained:
            early = EarlyStopping(
                monitor="val_kid",
                min_delta=0.001,
                patience=7,
            )
            lr_reduce = ReduceLROnPlateau(
                monitor="val_kid",
                factor=0.1,
                patience=3,
                verbose=1,
                min_lr=1e-9,
            )
            if plot_imgs:
                
                callbacks = [
                    tf.keras.callbacks.LambdaCallback(on_epoch_end=self.plot_images),
                    early,
                    lr_reduce
                    #checkpoint_callback,
                ]
            else:
                callbacks = [early, lr_reduce]
            
            tf_lr = transfer_learning_rate
            
            for i in range(23):
                self.compile(
                        optimizer=tf.keras.optimizers.AdamW(
                            learning_rate=tf_lr, weight_decay=weight_decay
                        ),
                        loss=keras.losses.mean_absolute_error,
                    )
                
                dot_graph = tf.keras.utils.model_to_dot(self.network)
                with open('networksummary.txt', 'w') as f:
                    self.network.summary(print_fn=lambda x: f.write(x + '\n'))

                with open('emanetworksummary.txt', 'w') as f:
                    self.ema_network.summary(print_fn=lambda x: f.write(x + '\n'))
                with open ('model.dot', 'w') as f:
                    f.write(dot_graph.to_string())
                
                tf.keras.utils.plot_model(
                    self.network,
                    to_file='network.png',
                    show_shapes=True,
                    show_layer_names=True,
                )
                tf.keras.utils.plot_model(
                    self.ema_network,
                    to_file='ema_network.png',
                    show_shapes=True,
                    show_layer_names=True,
                )

                import time
                #time.sleep(90000)
                flowers_dataset = prepare_dataset(f"train[{3.0*(i)}:{3.0*(i+1)}%]+test[{3.0*(i)}:{3.0*(i+1)}%]", image_size=self.image_size)
                val_flowers_dataset = prepare_dataset(f"train[{100-(1.0)*(i+1)}%:{100-(1.0)*(i)}]+test[{100-(1.0)*(i+1)}%:{100-(1.0)*(i)}]", image_size=self.image_size)

                # pixelwise mean absolute error is used as loss
                # calculate mean and variance of training dataset for normalization
                self.normalizer.adapt(flowers_dataset)
                
                self.img_name = "./newtf/PretrainedNetGeneration.png"
                self.fit(
                    flowers_dataset,
                    epochs=num_epochs_flowers,
                    validation_data=val_flowers_dataset,
                    callbacks=callbacks,
                    shuffle=True
                )
                tf_lr = tf_lr / 1.05

                del flowers_dataset
                del val_flowers_dataset


                if save:
                    self.network.save('newtf/diffusion_pretrained.h5')
                    self.ema_network.save('newtf/ema_diffusion_pretrained.h5')
                    #self.network.save_weights('./diffusion_pretrained/checkpoints/my_checkpoint')
        else:
            self.network = tf.keras.models.load_model('newtf/diffusion_pretrained.h5')
            self.ema_network = tf.keras.models.load_model('newtf/ema_diffusion_pretrained.h5')
            #self.network.load_weights('./diffusion_pretrained/checkpoints/my_checkpoint')
            print('Loaded pretrained model')

        self.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=learning_rate, weight_decay=weight_decay
                ),
                loss=keras.losses.mean_absolute_error,
            )
        
        

        #self.network.summary()
        #tf.keras.utils.plot_model(self.network, show_shapes=True, to_file="attention_unet.png")
        #self.ema_network.summary()

        for layer in self.network.layers[0:int(len(self.network.layers)/2)]:
            layer.trainable = False
            #print(layer.name)
        for layer in self.ema_network.layers[0:int(len(self.ema_network.layers)/2)]:
            layer.trainable = False

        #self.network.summary()
        #self.ema_network.summary()

        train_dataset = tf.data.Dataset.from_tensor_slices(list(np.asarray(train_dataset, dtype="float32") / 255.0))
        val_dataset = tf.data.Dataset.from_tensor_slices(list(np.asarray(val_dataset, dtype="float32") / 255.0))
            
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

        # run training and plot generated images periodically
        early = EarlyStopping(
                monitor="val_kid",
                min_delta=0.001,
                patience=15,
        )
        lr_reduce = ReduceLROnPlateau(
                monitor="val_kid",
                factor=0.2,
                patience=6,
                verbose=1,
                min_lr=1e-9,
            )
        if plot_imgs:
            
            callbacks = [
                tf.keras.callbacks.LambdaCallback(on_epoch_end=self.plot_images),
                early,
                lr_reduce
                #checkpoint_callback,
            ]
        else:
            callbacks = [early, lr_reduce]
        
        
        self.normalizer.adapt(train_dataset)
        if get_pretrained:
            if plot_imgs:
                print("Pretrained images generation")
                self.img_name = "./PretrainedNetGeneration.png"
                self.plot_images()

        if lamp:
            self.img_name = f"./newtf/{percent}/Lamps{culture}_{category}_imb={imb}.png"
        else:
            self.img_name = f"./newtf/{percent}/Carpets{culture}_{category}_imb={imb}.png"
        self.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
        )
        """
        for layer in self.network.layers[0:int(len(self.network.layers))]:
            layer.trainable = True
        for layer in self.ema_network.layers[0:int(len(self.ema_network.layers))]:
            layer.trainable = True

        # Fine tuning
        
        self.compile(
                optimizer=AdamW(
                    learning_rate=learning_rate/100, weight_decay=weight_decay
                ),
                loss=tf.keras.losses.mean_absolute_error,
            )

        self.fit(
            train_dataset,
            epochs=num_epochs//3,
            validation_data=val_dataset,
            callbacks=callbacks,

        )
        """
        tot = 0
        generated_images = []
        for i in range(n_images//batch_size +1):
            tot +=batch_size
            n_ = min((n_images-tot), batch_size)
            if n_>0:
                images = self.generate(
                    num_images=n_,
                    diffusion_steps=plot_diffusion_steps,
                )
                for img in images:
                    generated_images.append(img*255)

        generated_images = np.asarray(generated_images)

        return generated_images
        
        
