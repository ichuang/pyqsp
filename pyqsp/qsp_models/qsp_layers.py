import numpy as np
import tensorflow as tf
from tensorflow import keras


class QSP(keras.layers.Layer):
    """Parameterized quantum signal processing layer.

    The `QSP` layer implements the quantum signal processing circuit with trainable QSP angles.
    The input of the layer is/are theta(s) where x = cos(theta), and w(x) is X rotation in the QSP sequence.

    The output is the real part of the upper left element in the resulting unitary that describes the whole sequence.
    This is Re[P(x)] in the representation of the QSP unitary from Gilyen et al.

    Input is of the form:
    [[theta1], [theta2], ... ]

    Output is of the form:
    [[P(x1)], [P(x2)], ...]

    The layer requires the desired polynomial degree of P(x)

    """

    def __init__(self, poly_deg=0, measurement="z"):
        """
        Params
        ------
        poly_deg: The desired degree of the polynomial in the QSP sequence.
            the layer will be parameterized with poly_deg + 1 trainable phi.
        measurement :
            measurement basis using the Wx model, {"x", "z"}
        """
        super(QSP, self).__init__()
        self.poly_deg = poly_deg
        phi_init = tf.random_uniform_initializer(minval=0, maxval=np.pi)
        self.phis = tf.Variable(
            initial_value=phi_init(shape=(poly_deg + 1, 1), dtype=tf.float32),
            trainable=True,
        )
        self.measurement = measurement

    def call(self, th):
        batch_dim = tf.gather(tf.shape(th), 0)

        # tiled up X rotations (input W(x))
        px = tf.constant([[0.0, 1], [1, 0]], dtype=tf.complex64)
        px = tf.expand_dims(px, axis=0)
        px = tf.repeat(px, [batch_dim], axis=0)

        rot_x_arg = tf.complex(real=0.0, imag=th)
        rot_x_arg = tf.expand_dims(rot_x_arg, axis=1)
        rot_x_arg = tf.tile(rot_x_arg, [1, 2, 2])

        wx = tf.linalg.expm(tf.multiply(px, rot_x_arg))

        # tiled up Z rotations
        pz = tf.constant([[1.0, 0], [0, -1]], dtype=tf.complex64)
        pz = tf.expand_dims(pz, axis=0)
        pz = tf.repeat(pz, [batch_dim], axis=0)

        z_rotations = []
        for k in range(self.poly_deg + 1):
            phi = self.phis[k]
            rot_z_arg = tf.complex(real=0.0, imag=phi)
            rot_z_arg = tf.expand_dims(rot_z_arg, axis=0)
            rot_z_arg = tf.expand_dims(rot_z_arg, axis=0)
            rot_z_arg = tf.tile(rot_z_arg, [batch_dim, 2, 2])

            rz = tf.linalg.expm(tf.multiply(pz, rot_z_arg))
            z_rotations.append(rz)

        u = z_rotations[0]
        for rz in z_rotations[1:]:
            u = tf.matmul(u, wx)
            u = tf.matmul(u, rz)

        # assume we are interested in the real part of p(x) and the real part of q(x) in
        # the resulting qsp unitary
        if self.measurement == "z":
            return tf.math.real(u[:, 0, 0]), tf.math.imag(u[:, 0, 0])
        elif self.measurement == "x":
            return tf.math.real(u[:, 0, 0]), tf.math.imag(u[:, 0, 1])
        else:
            raise ValueError(
                "Invalid measurement basis: {}".format(self.measurement))


def construct_qsp_model(poly_deg, measurement="z"):
    """Helper function that compiles a QSP model with mean squared error and adam optimizer.

    Params
    ------
    poly_deg : int
        the desired degree of the polynomial in the QSP sequence.
    measurement :
        measurement basis using the Wx model, {"x", "z"}

    Returns
    -------
    Keras model
        a compiled keras model with trainable phis in a poly_deg QSP sequence.
    """
    theta_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="theta")
    qsp = QSP(poly_deg, measurement=measurement)
    real_parts = qsp(theta_input)
    model = tf.keras.Model(inputs=theta_input, outputs=real_parts)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    return model
