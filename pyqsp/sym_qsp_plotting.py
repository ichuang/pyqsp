import numpy as np
import sym_qsp_opt
import matplotlib.pyplot as plt

def main():
    coef = np.array([0.2,0.1,0.1,0.1,0.1])
    full_coef = np.zeros(2*len(coef))
    parity = 0

    # Generate full Chebyshev coefficients by padding with zeros.
    full_coef[parity::2] = coef
    # Compute the ideal polynomial from Chebyshev coefficients.
    desired_fun = lambda x: np.polynomial.chebyshev.chebval(x, full_coef)

    # Optimize to the desired function.
    (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_Solver(coef, parity)    
    print("phases: %s\nerror: %s\niter: %s\n"%(str(phases), str(err), str(total_iter)))

    # Plot achieved versus desired function.
    num_samples = 150
    samples = np.linspace(-1,1,num=num_samples)

    re_vals = qsp_seq_opt.gen_response_re(samples)
    im_vals = qsp_seq_opt.gen_response_im(samples)

    # TODO: Note the current requirement of a negative sign; check the reasoning.
    des_vals = -1*np.array(list(map(desired_fun, samples)))

    plt.plot(samples, re_vals, 'r', label="Real")
    plt.plot(samples, im_vals, 'g', label="Imag")
    plt.plot(samples, des_vals, 'b', label="Des")

    plt.ylabel("Test")
    plt.xlabel("Test")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()