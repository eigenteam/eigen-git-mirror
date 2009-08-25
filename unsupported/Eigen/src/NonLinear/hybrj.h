
template<typename FunctorType, typename Scalar=double>
class HybridNonLinearSolver
{
public:
    HybridNonLinearSolver(const FunctorType &_functor)
        : functor(_functor) {}

    int solve(
            Matrix< Scalar, Dynamic, 1 >  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );
    int solve(
            Matrix< Scalar, Dynamic, 1 >  &x,
            int &nfev, int &njev,
            Matrix< Scalar, Dynamic, 1 >  &diag,
            const int mode=1,
            const int maxfev = 1000,
            const Scalar factor = Scalar(100.),
            const Scalar xtol = ei_sqrt(epsilon<Scalar>()),
            const int nprint=0
            );

    Matrix< Scalar, Dynamic, 1 >  fvec;
    Matrix< Scalar, Dynamic, Dynamic > fjac;
    Matrix< Scalar, Dynamic, 1 >  R;
    Matrix< Scalar, Dynamic, 1 >  qtf;
private:
    const FunctorType &functor;
};



template<typename FunctorType, typename Scalar>
int HybridNonLinearSolver<FunctorType,Scalar>::solve(
        Matrix< Scalar, Dynamic, 1 >  &x,
        const Scalar tol
        )
{
    const int n = x.size();
    int info, nfev=0, njev=0;
    Matrix< Scalar, Dynamic, 1> diag;

    /* check the input parameters for errors. */
    if (n <= 0 || tol < 0.) {
        printf("HybridNonLinearSolver::solve() bad args : n,tol,...");
        return 0;
    }

    diag.setConstant(n, 1.);
    info = solve(
        x, 
        nfev, njev,
        diag,
        2,
        (n+1)*100,
        100.,
        tol
    );
    return (info==5)?4:info;
}



template<typename FunctorType, typename Scalar>
int HybridNonLinearSolver<FunctorType,Scalar>::solve(
        Matrix< Scalar, Dynamic, 1 >  &x,
        int &nfev,
        int &njev,
        Matrix< Scalar, Dynamic, 1 >  &diag,
        const int mode,
        const int maxfev,
        const Scalar factor,
        const Scalar xtol,
        const int nprint
        )
{
    const int n = x.size();
    Matrix< Scalar, Dynamic, 1 > wa1(n), wa2(n), wa3(n), wa4(n);

    fvec.resize(n);
    qtf.resize(n);
    R.resize( (n*(n+1))/2);
    fjac.resize(n, n);
    fvec.resize(n);

    /* Local variables */
    int i, j, l, iwa[1];
    Scalar sum;
    int sing;
    int iter;
    Scalar temp;
    int iflag;
    Scalar delta;
    int jeval;
    int ncsuc;
    Scalar ratio;
    Scalar fnorm;
    Scalar pnorm, xnorm, fnorm1;
    int nslow1, nslow2;
    int ncfail;
    Scalar actred, prered;
    int info;

    /* Function Body */
    info = 0;
    iflag = 0;
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || xtol < 0. || maxfev <= 0 || factor <= 0. )
        goto algo_end;
    if (mode == 2)
        for (j = 0; j < n; ++j)
            if (diag[j] <= 0.) goto algo_end;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    iflag = functor.f(x, fvec);
    nfev = 1;
    if (iflag < 0)
        goto algo_end;
    fnorm = fvec.stableNorm();

    /*     initialize iteration counter and monitors. */

    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

    /*     beginning of the outer loop. */

    while (true) {
        jeval = true;

        /* calculate the jacobian matrix. */

        iflag = functor.df(x, fjac);
        ++njev;
        if (iflag < 0)
            break;

        /* compute the qr factorization of the jacobian. */

        ei_qrfac<Scalar>(n, n, fjac.data(), fjac.rows(), false, iwa, 1, wa1.data(), wa2.data());

        /* on the first iteration and if mode is 1, scale according */
        /* to the norms of the columns of the initial jacobian. */

        if (iter == 1) {
            if (mode != 2)
                for (j = 0; j < n; ++j) {
                    diag[j] = wa2[j];
                    if (wa2[j] == 0.)
                        diag[j] = 1.;
                }

            /* on the first iteration, calculate the norm of the scaled x */
            /* and initialize the step bound delta. */

            wa3 = diag.cwise() * x;
            xnorm = wa3.stableNorm();
            delta = factor * xnorm;
            if (delta == 0.)
                delta = factor;
        }

        /* form (q transpose)*fvec and store in qtf. */

        qtf = fvec;
        for (j = 0; j < n; ++j)
            if (fjac(j,j) != 0.) {
                sum = 0.;
                for (i = j; i < n; ++i)
                    sum += fjac(i,j) * qtf[i];
                temp = -sum / fjac(j,j);
                for (i = j; i < n; ++i)
                    qtf[i] += fjac(i,j) * temp;
            }

        /* copy the triangular factor of the qr factorization into r. */

        sing = false;
        for (j = 0; j < n; ++j) {
            l = j;
            if (j)
                for (i = 0; i < j; ++i) {
                    R[l] = fjac(i,j);
                    l = l + n - i -1;
                }
            R[l] = wa1[j];
            if (wa1[j] == 0.)
                sing = true;
        }

        /* accumulate the orthogonal factor in fjac. */

        ei_qform<Scalar>(n, n, fjac.data(), fjac.rows(), wa1.data());

        /* rescale if necessary. */

        /* Computing MAX */
        if (mode != 2)
            diag = diag.cwise().max(wa2);

        /* beginning of the inner loop. */

        while (true) {
            /* if requested, call functor.f to enable printing of iterates. */

            if (nprint > 0) {
                iflag = 0;
                if ((iter - 1) % nprint == 0)
                    iflag = functor.debug(x, fvec, fjac);
                if (iflag < 0)
                    goto algo_end;
            }

            /* determine the direction p. */

            ei_dogleg<Scalar>(R, diag, qtf, delta, wa1);

            /* store the direction p and x + p. calculate the norm of p. */

            wa1 = -wa1;
            wa2 = x + wa1;
            wa3 = diag.cwise() * wa1;
            pnorm = wa3.stableNorm();

            /* on the first iteration, adjust the initial step bound. */

            if (iter == 1)
                delta = std::min(delta,pnorm);

            /* evaluate the function at x + p and calculate its norm. */

            iflag = functor.f(wa2, wa4);
            ++nfev;
            if (iflag < 0)
                goto algo_end;
            fnorm1 = wa4.stableNorm();

            /* compute the scaled actual reduction. */

            actred = -1.;
            if (fnorm1 < fnorm) /* Computing 2nd power */
                actred = 1. - ei_abs2(fnorm1 / fnorm);

            /* compute the scaled predicted reduction. */

            l = 0;
            for (i = 0; i < n; ++i) {
                sum = 0.;
                for (j = i; j < n; ++j) {
                    sum += R[l] * wa1[j];
                    ++l;
                }
                wa3[i] = qtf[i] + sum;
            }
            temp = wa3.stableNorm();
            prered = 0.;
            if (temp < fnorm) /* Computing 2nd power */
                prered = 1. - ei_abs2(temp / fnorm);

            /* compute the ratio of the actual to the predicted */
            /* reduction. */

            ratio = 0.;
            if (prered > 0.)
                ratio = actred / prered;

            /* update the step bound. */

            if (ratio < Scalar(.1)) {
                ncsuc = 0;
                ++ncfail;
                delta = Scalar(.5) * delta;
            } else {
                ncfail = 0;
                ++ncsuc;
                if (ratio >= Scalar(.5) || ncsuc > 1) /* Computing MAX */
                    delta = std::max(delta, pnorm / Scalar(.5));
                if (ei_abs(ratio - 1.) <= Scalar(.1)) {
                    delta = pnorm / Scalar(.5);
                }
            }

            /* test for successful iteration. */

            if (ratio >= Scalar(1e-4)) {
                /* successful iteration. update x, fvec, and their norms. */
                x = wa2;
                wa2 = diag.cwise() * x;
                fvec = wa4;
                xnorm = wa2.stableNorm();
                fnorm = fnorm1;
                ++iter;
            }

            /* determine the progress of the iteration. */

            ++nslow1;
            if (actred >= Scalar(.001))
                nslow1 = 0;
            if (jeval)
                ++nslow2;
            if (actred >= Scalar(.1))
                nslow2 = 0;

            /* test for convergence. */

            if (delta <= xtol * xnorm || fnorm == 0.)
                info = 1;
            if (info != 0)
                goto algo_end;

            /* tests for termination and stringent tolerances. */

            if (nfev >= maxfev)
                info = 2;
            /* Computing MAX */
            if (Scalar(.1) * std::max(Scalar(.1) * delta, pnorm) <= epsilon<Scalar>() * xnorm)
                info = 3;
            if (nslow2 == 5)
                info = 4;
            if (nslow1 == 10)
                info = 5;
            if (info != 0)
                goto algo_end;

            /* criterion for recalculating jacobian. */

            if (ncfail == 2)
                break;

            /* calculate the rank one modification to the jacobian */
            /* and update qtf if necessary. */

            for (j = 0; j < n; ++j) {
                sum = wa4.dot(fjac.col(j));
                wa2[j] = (sum - wa3[j]) / pnorm;
                wa1[j] = diag[j] * (diag[j] * wa1[j] / pnorm);
                if (ratio >= Scalar(1e-4))
                    qtf[j] = sum;
            }

            /* compute the qr factorization of the updated jacobian. */

            ei_r1updt<Scalar>(n, n, R.data(), R.size(), wa1.data(), wa2.data(), wa3.data(), &sing);
            ei_r1mpyq<Scalar>(n, n, fjac.data(), fjac.rows(), wa2.data(), wa3.data());
            ei_r1mpyq<Scalar>(1, n, qtf.data(), 1, wa2.data(), wa3.data());

            /* end of the inner loop. */

            jeval = false;
        }
        /* end of the outer loop. */
    }
algo_end:
    /*     termination, either normal or user imposed. */
    if (iflag < 0)
        info = iflag;
    if (nprint > 0)
        iflag = functor.debug(x, fvec, fjac);
    return info;
}

