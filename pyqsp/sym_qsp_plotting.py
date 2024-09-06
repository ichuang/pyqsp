import numpy as np
import sym_qsp_opt
import matplotlib.pyplot as plt
import scipy.special
import poly

def main():

    #######################################################
    #                                                     #
    #                  COSINE APPROX                      #
    #                                                     #
    #######################################################

    # Call existing methods to compute Jacobi-Anger expression for cosine.
    freq = 16
    pg = poly.PolyCosineTX()
    
    # Note new argument to use Chebyshev basis.
    pcoefs = pg.generate(tau=freq, epsilon=1e-12, chebyshev_basis=True)
    
    # Generate anonymous function (approx to cosine) using pcoefs.
    cos_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)
    
    # Initialize definite parity coefficients, and slice out nontrivial ones.
    parity = 0
    coef = pcoefs[parity::2]

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
    axs[0].plot(samples, des_vals, 'g', label="Ideal function")
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

    #######################################################
    #                                                     #
    #                   SINE APPROX                       #
    #                                                     #
    #######################################################

    # Call existing methods to compute Jacobi-Anger expression for sine.
    freq = 16
    pg = poly.PolySineTX()
    
    # Note new argument to use Chebyshev basis.
    pcoefs = pg.generate(tau=freq, epsilon=1e-12, chebyshev_basis=True)
    
    # Generate anonymous function (approx to cosine) using pcoefs.
    sin_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)
    
    # Initialize definite parity coefficients, and slice out nontrivial ones.
    full_coef = pcoefs
    parity = 1
    coef = full_coef[parity::2]

    # Anonymous function for ideal polynomial from Chebyshev coefficients.
    true_fun = lambda x: 0.5*np.sin(freq*x)

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
    sin_vals = np.array(list(map(sin_fun, samples)))

    # Generate simultaneous plots.
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle('Approximating sine with QSP to machine precision')

    # Standard plotting of relevant components.
    axs[0].plot(samples, im_vals, 'r', label="QSP poly")
    axs[0].plot(samples, sin_vals, 'b', label="Target poly")
    axs[0].plot(samples, des_vals, 'g', label="Ideal fun")
    # plt.plot(samples, re_vals, 'r', label="Real") # Unimportant real component.

    diff = np.abs(im_vals - sin_vals)
    true_diff = np.abs(des_vals - sin_vals)
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

    #######################################################
    #                                                     #
    #                 INVERSE APPROX                      #
    #                                                     #
    #######################################################

    # Generate inverse polynomial approximation
    pg = poly.PolyOneOverX()
    
    # Underlying parameters of inverse approximation.
    # We use return_scale=True for ease of plotting correct desired function.
    kappa=5
    epsilon=0.01
    pcoefs, scale = pg.generate(kappa=kappa, epsilon=epsilon, chebyshev_basis=True, return_scale=True)
    
    # Generate anonymous approximation and ideal function for scaled reciprocal.
    inv_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)
    ideal_fun = lambda x: scale*np.reciprocal(x)
    
    # Using odd parity and instantiating desired coefficeints.
    parity = 1
    coef = pcoefs[parity::2]

    # Optimize to the desired function using Newton solver.
    crit = 1e-12
    (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_Solver(coef, parity, crit=crit)    
    print("phase len: %s\nerror: %s\niter: %s\n"%(str(len(phases)), str(err), str(total_iter)))

    # Generate samples for plotting.
    num_samples = 400
    sample_0 = np.linspace(-1, -1.0/10000,num=num_samples)
    sample_1 = np.linspace(1.0/10000,1,num=num_samples)
    # Adding NaN between ranges to remove plotting artifacts.
    samples = np.concatenate((sample_0, [float('NaN')], sample_1))

    # Grab im part of QSP unitary top-left matrix element.
    im_vals = np.array(qsp_seq_opt.gen_response_im(samples))
    
    # Generate plotted values.
    approx_vals = np.array(list(map(inv_fun, samples)))
    # NOTE: For some reason this map casts to have an additional dimension.
    ideal_vals = np.array(list(map(ideal_fun, samples)))[:,0]

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('Approximating $1/x$ with QSP on $[-1,-1/5]\\cup[1/5, 1]$')

    # Plot achieved QSP protocol along with approximating polynomial.
    axs[0].plot(samples, approx_vals, 'r', label="Poly approx")
    axs[0].plot(samples, im_vals, 'g', label="Poly QSP")
    axs[0].plot(samples, ideal_vals, 'b', label="True function")

    # Plot difference between two on log-plot
    diff = np.abs(im_vals - approx_vals)
    approx_diff = np.abs(ideal_vals - approx_vals)
    
    # Plot QSP output polynomial versus desired polynomial, and error bound.
    axs[1].plot(samples, diff, 'r', label="Approx vs QSP")
    axs[1].plot(samples, [crit]*len(samples), 'y', label="QSP error limit")
    
    # Plot approximation versus ideal function, and error bound.
    axs[1].plot(samples, approx_diff, 'g', label="Approx vs ideal")
    axs[1].plot(samples, [epsilon]*len(samples), 'b', label="Approx error limit")
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

    axs[0].axvspan(-1.0/kappa, 1/kappa, alpha=0.1, color='black',lw=0)
    axs[1].axvspan(-1.0/kappa, 1/kappa, alpha=0.1, color='black',lw=0)

    plt.show()

    #######################################################
    #                                                     #
    #                   SIGN APPROX                       #
    #                                                     #
    #######################################################

    # Call existing methods to compute approximation to rect.
    freq = 16
    pg = poly.PolySign()
    
    # TODO: note that definition of PolySign has been changed to return bare pcoefs and not TargetPolynomial
    pcoefs, scale = pg.generate(
            degree=161,
            delta=25,
            chebyshev_basis=True, 
            cheb_samples=250,
            return_scale=True)
    # Cast from TargetPolynomial class bare Chebyshev coefficients if not using return_scale.
    # pcoefs = pcoefs.coef
    
    # Generate anonymous function (approx to cosine) using pcoefs.
    sign_fun = lambda x: np.polynomial.chebyshev.chebval(x, pcoefs)
    
    # Initialize definite parity coefficients, and slice out nontrivial ones.
    parity = 1
    bare_coefs = pcoefs
    coef = bare_coefs[parity::2]

    # Anonymous function for ideal polynomial from Chebyshev coefficients.
    true_fun = lambda x: scale * scipy.special.erf(x * 20)

    # Optimize to the desired function using Newton solver.
    (phases, err, total_iter, qsp_seq_opt) = sym_qsp_opt.newton_Solver(coef, parity, crit=1e-12)    
    print("phases: %s\nerror: %s\niter: %s\n"%(str(phases), str(err), str(total_iter)))

    # Plot achieved versus desired function over samples.
    num_samples = 400
    samples = np.linspace(-1,1,num=num_samples)

    # Compute real and imaginary parts of (0,0) matrix element.
    im_vals = np.array(qsp_seq_opt.gen_response_im(samples))

    # Map the desired function and achieved function over samples.
    des_vals = np.array(list(map(true_fun, samples)))[:,0] # Note shape casting.
    sign_vals = np.array(list(map(sign_fun, samples)))

    # Generate simultaneous plots.
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle('Approximating sign with QSP to machine precision')

    # Standard plotting of relevant components.
    axs[0].plot(samples, im_vals, 'r', label="QSP imag poly")
    axs[0].plot(samples, sign_vals, 'b', label="Target poly")
    axs[0].plot(samples, des_vals, 'g', label="Ideal fun")

    diff = np.abs(im_vals - sign_vals)
    true_diff = np.abs(des_vals - sign_vals)
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