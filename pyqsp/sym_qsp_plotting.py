import numpy as np
import sym_qsp_opt
import matplotlib.pyplot as plt
import poly

def main():

    # Call existing methods to compute Jacobi-Anger expression for cosine.
    freq = 16
    pg = poly.PolyCosineTX()
    
    # Note new argument to use Chebyshev basis.
    pcoefs = pg.generate(tau=freq, epsilon=1e-12, chebyshev_basis=True)
    
    # Generate anonymous function (approx to cosine) using pcoefs.
    cos_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)
    
    # Initialize definite parity coefficients, and slice out nontrivial ones.
    full_coef = pcoefs
    parity = 0
    coef = full_coef[parity::2]

    # Anonymous function for ideal polynomial from Chebyshev coefficients.
    true_fun = lambda x: 0.5*np.cos(freq*x)

    # Optimize to the desired function using Newton solver.
    (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_Solver(coef, parity, crit=1e-12)    
    print("phases: %s\nerror: %s\niter: %s\n"%(str(phases), str(err), str(total_iter)))

    # Plot achieved versus desired function over samples.
    num_samples = 400
    samples = np.linspace(-1,1,num=num_samples)

    # Compute real and imaginary parts of (0,0) matrix element.
    re_vals = np.array(qsp_seq_opt.gen_response_re(samples))
    im_vals = np.array(qsp_seq_opt.gen_response_im(samples))

    # Map the desired function and achieved function over samples.
    des_vals = np.array(list(map(true_fun, samples)))
    cos_vals = np.array(list(map(cos_fun, samples)))

    # Generate simultaneous plots.
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle('Approximating cosine with QSP to machine precision')

    # Standard plotting of relevant components.
    axs[0].plot(samples, im_vals, 'r', label="QSP poly")
    axs[0].plot(samples, cos_vals, 'b', label="Target poly")
    axs[0].plot(samples, des_vals, 'g', label="Ideal fun")
    # plt.plot(samples, re_vals, 'r', label="Real") # Unimportant real component.

    diff = np.abs(im_vals - cos_vals)
    true_diff = np.abs(des_vals - cos_vals)
    total_diff = np.abs(im_vals - des_vals)
    axs[1].plot(samples, diff, 'r', label="Approx vs QSP")
    axs[1].plot(samples, true_diff, 'g', label="Approx vs true")
    axs[1].plot(samples, total_diff, 'b', label="QSP vs true")
    axs[1].set_yscale('log')

    # Set axis limits and quality of life features.
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])
    axs[0].set_ylabel("Component value")
    axs[1].set_ylabel("Absolute error")
    axs[1].set_xlabel('Input signal')

    # Further cosmetic alterations
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    plt.show()

if __name__ == '__main__':
    main()