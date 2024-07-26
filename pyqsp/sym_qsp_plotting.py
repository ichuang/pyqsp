import numpy as np
import sym_qsp_opt
import matplotlib.pyplot as plt
import poly

def main():

    pg = poly.PolyOneOverX()
    pcoefs = pg.generate(4, 0.05, return_coef=True, ensure_bounded=True)
    # TODO: temporary hack! Change from poly convention back to cheb.
    pcoefs = np.polynomial.chebyshev.poly2cheb(pcoefs)
    inverse_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)

    # Generating some random Chebyshev decomposition.
    coef = np.array(10*[0.05])
    full_coef = np.zeros(2*len(coef))
    parity = 0

    # Generate full Chebyshev coefficients by padding with zeros.
    full_coef[parity::2] = coef

    """
    NOTE THIS: Here we temporarily bypass the above coefficients to try to achieve the inverse function
    """
    full_coef = pcoefs
    parity = 1
    coef = full_coef[parity::2]

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

    # TODO: Can go back to target real component later.
    des_vals = np.array(list(map(desired_fun, samples)))
    inv_vals = np.array(list(map(inverse_fun, samples)))

    # plt.plot(samples, re_vals, 'r', label="Real") # Currently unimportant
    plt.plot(samples, im_vals, 'g', label="Imag")
    plt.plot(samples, des_vals, 'b', label="Input poly")
    plt.plot(samples, inv_vals, 'y', label="Inverse")

    ax = plt.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    plt.ylabel("Matrix component value")
    plt.xlabel("Signal")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    main()