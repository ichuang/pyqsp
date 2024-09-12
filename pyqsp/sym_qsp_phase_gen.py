import numpy as np
import sym_qsp_opt
import matplotlib.pyplot as plt
import scipy.special
import poly
import angle_sequence

"""
A series of tests for what will eventually be the command-line callable
QuantumSignalProcessingPhases method with the 'sym_qsp' flag enabled.
"""
def main():

    # Generate the second Chebyshev polynomial
    test_poly = np.polynomial.chebyshev.Chebyshev([0, 0, 0, 0, 0, 1])

    (full_phases, reduced_phases, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly=test_poly,
        eps=1e-4,
        suc=1-1e-4,
        tolerance=1e-6,
        method='sym_qsp',
        chebyshev_basis=True
    )

    print("Computed phases")
    print(full_phases)
    print(reduced_phases)
    print(parity)

    ############################################################

    # Generate the second Chebyshev polynomial with array/list
    test_poly = [0, 0, 0, 0, 0, 1]

    (full_phases, reduced_phases, parity) = angle_sequence.QuantumSignalProcessingPhases(
        poly=test_poly,
        eps=1e-4,
        suc=1-1e-4,
        tolerance=1e-6,
        method='sym_qsp',
        chebyshev_basis=True
    )

    print("\nComputed phases (from list)")
    print(full_phases)
    print(reduced_phases)
    print(parity)


if __name__ == '__main__':
    main()
