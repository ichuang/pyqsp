import os
import unittest

import numpy as np
import tensorflow as tf

from pyqsp import qsp_models

class Test_qsp_models(unittest.TestCase):
    '''
    Unit tests for qsp_models component of pyqsp
    '''
    def setUp(self):
        '''
        Do imports here, so that this module can be entirely optional
        '''
        if self.is_enabled():
            pass
        else:
            print(
                "[pyqsp.test] Skipping qsp_model tests: export PYQSP_TEST_QSP_MODELS=1 to enable these tests")

    def is_enabled(self):
        enabled = 'PYQSP_TEST_QSP_MODELS' in os.environ
        return enabled

    def xtest_hsim1(self):
        if not self.is_enabled():
            return
        t = 3
        poly_deg = 6
        def f(x): return np.cos(t * x)
        model = qsp_models.construct_qsp_model(poly_deg)

        # The intput theta training values
        th_in = np.arange(0, np.pi, np.pi / 50)
        th_in = tf.reshape(th_in, (th_in.shape[0], 1))

        # The desired real part of p(x) which is the upper left value in the unitary of the qsp sequence
        # We also want that Re[q(x)] = 0
        expected_outputs = [f(np.cos(th_in)), np.zeros(th_in.shape)]

        model = qsp_models.construct_qsp_model(poly_deg)
        history = model.fit(x=th_in, y=expected_outputs,
                            epochs=1000, verbose=0)
        # plot_loss(history)
        # plot_qsp_response(f, model)
        response, all_th, circuit_px, circuit_qx = qsp_models.compute_qsp_response(
            model, return_all=True)
        desired = f(np.cos(all_th))
        assert abs(response - desired).mean() < 0.1

    def xtest_mpinverse(self):
        if not self.is_enabled():
            return

        d = 1 / 16
        e = 1 / 16
        k = 2 / d
        poly_deg = int((np.log(1 / e) / d))
        b = np.ceil(k * k * np.log(k / e))
        # odd polynomials for now
        poly_deg = poly_deg if (np.mod(poly_deg, 2) == 1) else (poly_deg + 1)

        # approximation to inverse function d/2x
        def f(x): return np.where(x != 0, d /
                                  2 * (1 - (1 - x ** 2) ** b) / x, 0)

        # The intput theta training values
        th_in = np.arange(0, np.pi, np.pi / 30)
        th_in = tf.reshape(th_in, (th_in.shape[0], 1))

        # The desired real part of p(x) which is the upper left value in the
        # unitary of the qsp sequence
        expected_outputs = [f(np.cos(th_in)), np.zeros(th_in.shape[0])]
        model = qsp_models.construct_qsp_model(poly_deg)
        history = model.fit(x=th_in, y=expected_outputs,
                            epochs=5000, verbose=0)
        # plot_loss(history)
        # plot_qsp_response(f, model)
        response, all_th, circuit_px, circuit_qx = qsp_models.compute_qsp_response(
            model, return_all=True)
        desired = f(np.cos(all_th))
        assert abs(response - desired).mean() < 0.1

    def test_ampamp1(self):
        if not self.is_enabled():
            return
        poly_deg = 19

        # approximation to inverse function d/2x
        def f(x): return np.where(x < 0, -1, np.where(x > 0, 1, 0))

        # The intput theta training values
        th_in = np.arange(0, np.pi, np.pi / 30)
        th_in = tf.reshape(th_in, (th_in.shape[0], 1))

        # The desired real part of p(x) which is the upper left value in the
        # unitary of the qsp sequence
        expected_outputs = [f(np.cos(th_in)), np.zeros(th_in.shape[0])]

        model = qsp_models.construct_qsp_model(poly_deg)
        history = model.fit(x=th_in, y=expected_outputs,
                            epochs=5000, verbose=0)
        response, all_th, circuit_px, circuit_qx = qsp_models.compute_qsp_response(
            model, return_all=True)
        desired = f(np.cos(all_th))
        assert abs(response - desired).mean() < 0.1
