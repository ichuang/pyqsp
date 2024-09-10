import numbers

import cirq
import numpy as np
import sympy
import tensorflow as tf

import tensorflow_quantum as tfq
from tensorflow_quantum.core.ops import tfq_unitary_op
from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python.layers.circuit_executors import (
    expectation, input_checks, sampled_expectation)


class HybridControlledPQC(tf.keras.layers.Layer):
    """Hybrid Controlled Parametrized Quantum Circuit (PQC) Layer."""

    def __init__(self,
                 model_circuit,
                 operators,
                 *,
                 controlled_symbol_names=None,
                 native_symbol_names=None,
                 repetitions=None,
                 backend=None,
                 initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
                 regularizer=None,
                 constraint=None,
                 differentiator=None,
                 **kwargs):
        """Instantiate this layer.
        Create a layer that will output expectation values of the given
        operators when fed quantum data to it's input layer. This layer will
        take two input tensors, one representing a quantum data source (these
        circuits must not contain any symbols) and the other representing
        control parameters for the model circuit that gets appended to the
        datapoints.
        model_circuit: `cirq.Circuit` containing `sympy.Symbols` that will be
            used as the model which will be fed quantum data inputs.
        operators: `cirq.PauliSum` or Python `list` of `cirq.PauliSum` objects
            used as observables at the end of the model circuit.
        repetitions: Optional Python `int` indicating how many samples to use
            when estimating expectation values. If `None` analytic expectation
            calculation is used.
        backend: Optional Backend to use to simulate states. Defaults to
            the native TensorFlow simulator (None), however users may also
            specify a preconfigured cirq simulation object to use instead.
            If a cirq object is given it must inherit `cirq.SimulatesFinalState`
            if `sampled_based` is True or it must inherit `cirq.Sampler` if
            `sample_based` is False.
        differentiator: Optional `tfq.differentiator` object to specify how
            gradients of `model_circuit` should be calculated.
        """
        super().__init__(**kwargs)
        # Ingest model_circuit.
        if not isinstance(model_circuit, cirq.Circuit):
            raise TypeError("model_circuit must be a cirq.Circuit object."
                            " Given: ".format(model_circuit))
        self._symbols_list = list(
            sorted(util.get_circuit_symbols(model_circuit)))

#         self._native_symbols_list = list(sorted(controlled_symbol_names))
#         self._controlled_symbols_list = list(sorted(native_symbol_names))
#         self._native_symbols = tf.constant([str(x) for x in self._native_symbols_list])
#         self._controlled_symbols = tf.constant([str(x) for x in self._controlled_symbols_list])

        self._symbols = tf.constant(
            [str(x) for x in native_symbol_names + controlled_symbol_names])

        self._circuit = util.convert_to_tensor([model_circuit])

        if len(self._symbols_list) == 0:
            raise ValueError("model_circuit has no sympy.Symbols. Please "
                             "provide a circuit that contains symbols so "
                             "that their values can be trained.")

        # Ingest operators.
        if isinstance(operators, (cirq.PauliString, cirq.PauliSum)):
            operators = [operators]

        if not isinstance(operators, (list, np.ndarray, tuple)):
            raise TypeError("operators must be a cirq.PauliSum or "
                            "cirq.PauliString, or a list, tuple, "
                            "or np.array containing them. "
                            "Got {}.".format(type(operators)))
        if not all([
                isinstance(op, (cirq.PauliString, cirq.PauliSum))
                for op in operators
        ]):
            raise TypeError("Each element in operators to measure "
                            "must be a cirq.PauliString"
                            " or cirq.PauliSum")

        self._operators = util.convert_to_tensor([operators])

        # Ingest and promote repetitions.
        self._analytic = False
        if repetitions is None:
            self._analytic = True

        if not self._analytic and not isinstance(
                repetitions, numbers.Integral):
            raise TypeError("repetitions must be a positive integer value."
                            " Given: ".format(repetitions))

        if not self._analytic and repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")

        if not self._analytic:
            self._repetitions = tf.constant(
                [[repetitions for _ in range(len(operators))]],
                dtype=tf.dtypes.int32)

        if not isinstance(
                backend,
                cirq.Sampler) and repetitions is not None and backend is not None:
            raise TypeError("provided backend does not inherit cirq.Sampler "
                            "and repetitions!=None. Please provide a backend "
                            "that inherits cirq.Sampler or set "
                            "repetitions=None.")

        if not isinstance(backend, cirq.SimulatesFinalState
                          ) and repetitions is None and backend is not None:
            raise TypeError("provided backend does not inherit "
                            "cirq.SimulatesFinalState and repetitions=None. "
                            "Please provide a backend that inherits "
                            "cirq.SimulatesFinalState.")

        # Ingest backend and differentiator.
        if self._analytic:
            self._layer = expectation.Expectation(
                backend=backend, differentiator=differentiator)
        else:
            self._layer = sampled_expectation.SampledExpectation(
                backend=backend, differentiator=differentiator)

        self._append_layer = elementary.AddCircuit()

        # create weights for only native symbols

        if not all(
                name in self._symbols_list for name in controlled_symbol_names):
            raise ValueError(
                "model_circuit does not contain all controlled symbol names ")

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        self._native_symbol_values = self.add_weight(
            'parameters',
            shape=(
                len(native_symbol_names),
            ),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint,
            dtype=tf.float32,
            trainable=True)

    @property
    def symbols(self):
        """The symbols that are managed by this layer (in-order).
        Note: `symbols[i]` indicates what symbol name the managed variables in
            this layer map to.
        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def symbol_values(self):
        """Returns a Python `dict` containing symbol name, value pairs.
        Returns:
            Python `dict` with `str` keys and `float` values representing
                the current symbol values.
        """
        return dict(zip(self.symbols, self.get_weights()[0]))

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, controlled_symbol_values):
        """Keras call function."""
        circuit_batch_dim = tf.gather(tf.shape(controlled_symbol_values), 0)
        tiled_up_model = tf.tile(self._circuit, [circuit_batch_dim])
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])
        tiled_up_native_symbol_values = tf.tile(
            [self._native_symbol_values], [circuit_batch_dim, 1])
        symbol_values = tf.concat(
            [tiled_up_native_symbol_values, controlled_symbol_values], 1)
        # tiled_up_parameters = tf.tile(symbol_values, [circuit_batch_dim, 1])

        if self._analytic:
            return self._layer(tiled_up_model,
                               symbol_names=self._symbols,
                               symbol_values=symbol_values,
                               operators=tiled_up_operators)
        else:
            tiled_up_repetitions = tf.tile(self._repetitions,
                                           [circuit_batch_dim, 1])
            return self._layer(tiled_up_model,
                               symbol_names=self._symbols,
                               symbol_values=symbol_values,
                               operators=tiled_up_operators,
                               repetitions=tiled_up_repetitions)


class Unitary(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """Instantiate a Unitary Layer.
        Create a layer that will calculate circuit unitary matrices and output
        them into the TensorFlow graph given a correct set of inputs.
        """
        super().__init__(**kwargs)
        self.unitary_op = tfq_unitary_op.get_unitary_op()
        self._w = None

    @tf.function
    def call(self, inputs,
             *,
             symbol_names=None,
             symbol_values=None,
             initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi)):
        """Keras call function.
        Input options:
            `inputs`, `symbol_names`, `symbol_values`:
                see `input_checks.expand_circuits`
        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, <size of state>, <size of state>]
                    or
                [number of circuits, <size of state>, <size of state>]
        """

        values_empty = False
        if symbol_values is None:
            values_empty = True

        inputs, symbol_names, symbol_values = input_checks.expand_circuits(
            inputs, symbol_names, symbol_values)

        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)

        if values_empty:
            # No symbol_values were provided. So we assume the user wants us
            # to create and manage variables for them. We will do so by
            # creating a weights variable and tiling it up to appropriate
            # size of [batch, num_symbols].

            if self._w is None:
                # don't re-add variable.
                self._w = self.add_weight(name='circuit_learnable_parameters',
                                          shape=symbol_names.shape,
                                          initializer=initializer)

            symbol_values = tf.tile(tf.expand_dims(self._w, axis=0),
                                    tf.stack([circuit_batch_dim, 1]))

        unitary = self.unitary_op(inputs, symbol_names, symbol_values)
        return unitary.to_tensor()


class QSP(tf.keras.layers.Layer):
    """ QSP for """

    def __init__(self, poly_deg=0, **kwargs):
        super().__init__(**kwargs)
        self.q = cirq.GridQubit(0, 0)
        self.poly_deg = poly_deg
        self.symbol_names = [sympy.Symbol(f'phi{k}') for k in range(
            poly_deg + 1)] + [sympy.Symbol(f'th')]

        initializer = tf.keras.initializers.RandomUniform(0, 2 * np.pi)

        self.phi = self.add_weight(name='circuit_learnable_parameters',
                                   shape=(poly_deg + 1,),
                                   initializer=initializer)

    @tf.function
    def call(self, theta_inp):
        """Keras call function.
        Input options:
            `inputs`, `symbol_names`, `symbol_values`:
                see `input_checks.expand_circuits`
        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, <size of state>, <size of state>]
                    or
                [number of circuits, <size of state>, <size of state>]
        """

        wx = cirq.Circuit(cirq.rx(2 * theta_inp))
        self.rot_zs = [cirq.Circuit(cirq.rz(2 * self.phi[k])(self.q))
                       for k in range(self.poly_deg)]

        full_circuit = self.rot_zs[0]
        full_circuit_test = cirq.Circuit(
            cirq.rz(2 * self.symbol_names[0])(self.q),
            cirq.rx(2 * self.symbol_names[-1])(self.q),
            cirq.rz(2 * self.symbol_names[0])(self.q)
        )

        phi_values = tf.expand_dims(self.phi, axis=0)
        symbol_values = tf.expand_dims(tf.concat([self.phi, [4]], 0), 0)

        tensor_full_circuit_test = tfq.convert_to_tensor([full_circuit_test])
        tfq.from_tensor(tfq.resolve_parameters(
            tensor_full_circuit_test, self.symbol_names, symbol_values))

        return full_circuit_test.unitary()[0, 0]
