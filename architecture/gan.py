from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class GANModel:
    def __init__(self, gen, disc):
        self.gen = gen
        self.disc = disc

    def get_gan_model(self):
        self.disc.trainable = False
        gan_output = self.disc(self.gen.output)

        model = Model(self.gen.input, gan_output)
        opt = Adam(lr=0.0001, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)

        return model
