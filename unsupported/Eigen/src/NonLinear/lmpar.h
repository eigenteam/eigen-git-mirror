
template <typename Scalar>
void ei_lmpar(
        Matrix< Scalar, Dynamic, Dynamic > &r__,
        const VectorXi &ipvt,
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
    int jm1, jp1;
    Scalar sum, parc, parl;
    int iter;
    Scalar temp, paru;
    int nsing;
    Scalar gnorm;
    Scalar dxnorm;


    /* Function Body */
    const Scalar dwarf = std::numeric_limits<Scalar>::min();
    const int n = r__.cols();
    assert(n==diag.size());
    assert(n==qtb.size());
    assert(n==x.size());

    Matrix< Scalar, Dynamic, 1 >  wa1(n), wa2(n);

    /*     compute and store in x the gauss-newton direction. if the */
    /*     jacobian is rank-deficient, obtain a least squares solution. */

    nsing = n-1;
    for (j = 0; j < n; ++j) {
        wa1[j] = qtb[j];
        if (r__(j,j) == 0. && nsing == n-1)
            nsing = j - 1;
        if (nsing < n-1) 
            wa1[j] = 0.;
    }
    for (k = 0; k <= nsing; ++k) {
        j = nsing - k;
        wa1[j] /= r__(j,j);
        temp = wa1[j];
        jm1 = j - 1;
        for (i = 0; i <= jm1; ++i)
            wa1[i] -= r__(i,j) * temp;
    }

    for (j = 0; j < n; ++j) {
        l = ipvt[j]-1;
        x[l] = wa1[j];
    }

    /*     initialize the iteration counter. */
    /*     evaluate the function at the origin, and test */
    /*     for acceptance of the gauss-newton direction. */

    iter = 0;
    for (j = 0; j < n; ++j) {
        wa2[j] = diag[j] * x[j];
        /* L70: */
    }
    dxnorm = wa2.blueNorm();
    fp = dxnorm - delta;
    if (fp <= Scalar(0.1) * delta) {
        goto L220;
    }

    /*     if the jacobian is not rank deficient, the newton */
    /*     step provides a lower bound, parl, for the zero of */
    /*     the function. otherwise set this bound to zero. */

    parl = 0.;
    if (nsing < n-1) {
        goto L120;
    }
    for (j = 0; j < n; ++j) {
        l = ipvt[j]-1;
        wa1[j] = diag[l] * (wa2[l] / dxnorm);
    }
    for (j = 0; j < n; ++j) {
        sum = 0.;
        jm1 = j - 1;
        for (i = 0; i <= jm1; ++i) 
            sum += r__(i,j) * wa1[i];
        wa1[j] = (wa1[j] - sum) / r__(j,j);
    }
    temp = wa1.blueNorm();
    parl = fp / delta / temp / temp;
L120:

    /*     calculate an upper bound, paru, for the zero of the function. */

    for (j = 0; j < n; ++j) {
        sum = 0.;
        for (i = 0; i <= j; ++i) {
            sum += r__(i,j) * qtb[i];
            /* L130: */
        }
        l = ipvt[j]-1;
        wa1[j] = sum / diag[l];
        /* L140: */
    }
    gnorm = wa1.stableNorm();
    paru = gnorm / delta;
    if (paru == 0.) {
        paru = dwarf / std::min(delta,Scalar(0.1));
    }

    /*     if the input par lies outside of the interval (parl,paru), */
    /*     set par to the closer endpoint. */

    par = std::max(par,parl);
    par = std::min(par,paru);
    if (par == 0.) {
        par = gnorm / dxnorm;
    }

    /*     beginning of an iteration. */

L150:
    ++iter;

    /*        evaluate the function at the current value of par. */

    if (par == 0.) {
        /* Computing MAX */
        par = std::max(dwarf,Scalar(.001) * paru);
    }
    temp = ei_sqrt(par);
    for (j = 0; j < n; ++j) {
        wa1[j] = temp * diag[j];
        /* L160: */
    }
    ei_qrsolv<Scalar>(n, r__.data(), r__.rows(), ipvt.data(), wa1.data(), qtb.data(), x.data(), sdiag.data(), wa2.data());
    for (j = 0; j < n; ++j) {
        wa2[j] = diag[j] * x[j];
        /* L170: */
    }
    dxnorm = wa2.blueNorm();
    temp = fp;
    fp = dxnorm - delta;

    /*        if the function is small enough, accept the current value */
    /*        of par. also test for the exceptional cases where parl */
    /*        is zero or the number of iterations has reached 10. */

    if (ei_abs(fp) <= Scalar(0.1) * delta || (parl == 0. && fp <= temp && temp < 0.) ||
            iter == 10) {
        goto L220;
    }

    /*        compute the newton correction. */

    for (j = 0; j < n; ++j) {
        l = ipvt[j]-1;
        wa1[j] = diag[l] * (wa2[l] / dxnorm);
        /* L180: */
    }
    for (j = 0; j < n; ++j) {
        wa1[j] /= sdiag[j];
        temp = wa1[j];
        jp1 = j + 1;
        for (i = jp1; i < n; ++i)
            wa1[i] -= r__(i,j) * temp;
    }
    temp = wa1.blueNorm();
    parc = fp / delta / temp / temp;

    /*        depending on the sign of the function, update parl or paru. */

    if (fp > 0.) {
        parl = std::max(parl,par);
    }
    if (fp < 0.) {
        paru = std::min(paru,par);
    }

    /*        compute an improved estimate for par. */

    /* Computing MAX */
    par = std::max(parl,par+parc);

    /*        end of an iteration. */

    goto L150;
L220:

    /*     termination. */

    if (iter == 0) {
        par = 0.;
    }
    return;

    /*     last card of subroutine lmpar. */

} /* lmpar_ */

