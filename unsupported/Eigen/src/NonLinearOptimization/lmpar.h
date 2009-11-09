
template <typename Scalar>
void ei_lmpar(
        Matrix< Scalar, Dynamic, Dynamic > &r,
        VectorXi &ipvt, // TODO : const once ipvt mess fixed
        const Matrix< Scalar, Dynamic, 1 >  &diag,
        const Matrix< Scalar, Dynamic, 1 >  &qtb,
        Scalar delta,
        Scalar &par,
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &sdiag)
{
    /* Local variables */
    int i, j, k, l;
    Scalar fp;
    Scalar sum, parc, parl;
    int iter;
    Scalar temp, paru;
    int nsing;
    Scalar gnorm;
    Scalar dxnorm;


    /* Function Body */
    const Scalar dwarf = std::numeric_limits<Scalar>::min();
    const int n = r.cols();
    assert(n==diag.size());
    assert(n==qtb.size());
    assert(n==x.size());

    Matrix< Scalar, Dynamic, 1 >  wa1(n), wa2(n);

    /* compute and store in x the gauss-newton direction. if the */
    /* jacobian is rank-deficient, obtain a least squares solution. */

    nsing = n-1;
    for (j = 0; j < n; ++j) {
        wa1[j] = qtb[j];
        if (r(j,j) == 0. && nsing == n-1)
            nsing = j - 1;
        if (nsing < n-1) 
            wa1[j] = 0.;
    }
    for (k = 0; k <= nsing; ++k) {
        j = nsing - k;
        wa1[j] /= r(j,j);
        temp = wa1[j];
        for (i = 0; i < j ; ++i)
            wa1[i] -= r(i,j) * temp;
    }

    for (j = 0; j < n; ++j) {
        l = ipvt[j];
        x[l] = wa1[j];
    }

    /* initialize the iteration counter. */
    /* evaluate the function at the origin, and test */
    /* for acceptance of the gauss-newton direction. */

    iter = 0;
    wa2 = diag.cwise() * x;
    dxnorm = wa2.blueNorm();
    fp = dxnorm - delta;
    if (fp <= Scalar(0.1) * delta) {
        par = 0;
        return;
    }

    /* if the jacobian is not rank deficient, the newton */
    /* step provides a lower bound, parl, for the zero of */
    /* the function. otherwise set this bound to zero. */

    parl = 0.;
    if (nsing >= n-1) {
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
        }
        for (j = 0; j < n; ++j) {
            sum = 0.;
            for (i = 0; i < j; ++i) 
                sum += r(i,j) * wa1[i];
            wa1[j] = (wa1[j] - sum) / r(j,j);
        }
        temp = wa1.blueNorm();
        parl = fp / delta / temp / temp;
    }

    /* calculate an upper bound, paru, for the zero of the function. */

    for (j = 0; j < n; ++j) {
        sum = 0.;
        for (i = 0; i <= j; ++i)
            sum += r(i,j) * qtb[i];
        l = ipvt[j];
        wa1[j] = sum / diag[l];
    }
    gnorm = wa1.stableNorm();
    paru = gnorm / delta;
    if (paru == 0.)
        paru = dwarf / std::min(delta,Scalar(0.1));

    /* if the input par lies outside of the interval (parl,paru), */
    /* set par to the closer endpoint. */

    par = std::max(par,parl);
    par = std::min(par,paru);
    if (par == 0.)
        par = gnorm / dxnorm;

    /* beginning of an iteration. */

    while (true) {
        ++iter;

        /* evaluate the function at the current value of par. */

        if (par == 0.)
            par = std::max(dwarf,Scalar(.001) * paru); /* Computing MAX */

        temp = ei_sqrt(par);
        wa1 = temp * diag;

        ipvt.cwise()+=1; // qrsolv() expects the fortran convention (as qrfac provides)
        ei_qrsolv<Scalar>(n, r.data(), r.rows(), ipvt.data(), wa1.data(), qtb.data(), x.data(), sdiag.data(), wa2.data());
        ipvt.cwise()-=1;

        wa2 = diag.cwise() * x;
        dxnorm = wa2.blueNorm();
        temp = fp;
        fp = dxnorm - delta;

        /* if the function is small enough, accept the current value */
        /* of par. also test for the exceptional cases where parl */
        /* is zero or the number of iterations has reached 10. */

        if (ei_abs(fp) <= Scalar(0.1) * delta || (parl == 0. && fp <= temp && temp < 0.) || iter == 10)
            break;

        /* compute the newton correction. */

        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            wa1[j] = diag[l] * (wa2[l] / dxnorm);
            /* L180: */
        }
        for (j = 0; j < n; ++j) {
            wa1[j] /= sdiag[j];
            temp = wa1[j];
            for (i = j+1; i < n; ++i)
                wa1[i] -= r(i,j) * temp;
        }
        temp = wa1.blueNorm();
        parc = fp / delta / temp / temp;

        /* depending on the sign of the function, update parl or paru. */

        if (fp > 0.)
            parl = std::max(parl,par);
        if (fp < 0.)
            paru = std::min(paru,par);

        /* compute an improved estimate for par. */

        /* Computing MAX */
        par = std::max(parl,par+parc);

        /* end of an iteration. */

    }

    /* termination. */

    if (iter == 0)
        par = 0.;
    return;
}

