import time

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from architecture.generator import Generator
from architecture.discriminator import Discriminator

from data_read.read import Dataset

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 10
EMBED_SIZE = 512
EPOCHS = 500
BATCH_SIZE = 32


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_irrelevant_data, cycled_irrelevant_data):
    loss1 = tf.reduce_mean(tf.abs(real_irrelevant_data - cycled_irrelevant_data))
    return LAMBDA * loss1


def identity_loss(real_irrelevant_data, same_irrelevant_data):
    loss = tf.reduce_mean(tf.abs(real_irrelevant_data - same_irrelevant_data))
    return LAMBDA * 0.5 * loss


def checkpoint(generator_g, generator_f, discriminator_x, discriminator_y,
               generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer,
               discriminator_y_optimizer):
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f,
                               discriminator_x=discriminator_x,
                               discriminator_y=discriminator_y,
                               generator_g_optimizer=generator_g_optimizer,
                               generator_f_optimizer=generator_f_optimizer,
                               discriminator_x_optimizer=discriminator_x_optimizer,
                               discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return ckpt_manager


# fetch processed data
data = Dataset('/home/abhash/PycharmProjects/shitty_to_crisp_question/data/squad_v2.json',
               '/home/abhash/PycharmProjects/shitty_to_crisp_question/data/News_Category_Dataset_v2.json')
data.prepare_dataset()

irrelevant_dataset = data.encoded_irrelevant_questions_set
relevant_dataset = data.encoded_relevant_questions_set
max_len_question = data.max_len_question

# create optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator(EMBED_SIZE, max_len_question)
generator_g = generator.get_model()

# reinitializing
generator = Generator(EMBED_SIZE, max_len_question)
generator_f = generator.get_model()

discriminator = Discriminator(EMBED_SIZE, max_len_question)
discriminator_x = discriminator.get_model()

# reinitializing
discriminator = Discriminator(EMBED_SIZE, max_len_question)
discriminator_y = discriminator.get_model()

# test samples
test_encoded_samples = data.test_sentences


def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


ckpt_manager = checkpoint(
    generator_g, generator_f, discriminator_x, discriminator_y,
    generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer,
    discriminator_y_optimizer
)


total_train_data = len(irrelevant_dataset)
for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for step in range(int(total_train_data / BATCH_SIZE)):
        real_x = irrelevant_dataset[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE]
        real_y = relevant_dataset[step * BATCH_SIZE: step * BATCH_SIZE + BATCH_SIZE]

        # type cast from tf.float64 to tf.float32
        real_x = tf.cast(real_x, tf.float32)
        real_y = tf.cast(real_y, tf.float32)

        total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss = train_step(real_x, real_y)
        if step % 5 == 0:
            print(f"total_gen_g_loss: {total_gen_g_loss}, total_gen_f_loss: {total_gen_f_loss}, disc_x_loss: {disc_x_loss}, disc_y_loss: {disc_y_loss}")
    if n % 10 == 0:
        print('.', end='')
    n += 1

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

# test
targets = []
predicted = []
for test_x, test_y in test_encoded_samples:
    fake_y = generator_g(test_x, training=False)
    fake_y = np.squeeze(fake_y)
    print(cosine_similarity(fake_y, test_y[0]))
    targets.extend(test_y[0])
    predicted.extend(fake_y)
    print('#############################')

sim = cosine_similarity(predicted, targets)
#print(sim)
