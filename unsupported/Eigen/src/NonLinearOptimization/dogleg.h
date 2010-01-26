
template <typename Scalar>
void ei_dogleg(
        const Matrix< Scalar, Dynamic, Dynamic >  &qrfac,
        const Matrix< Scalar, Dynamic, 1 >  &diag,
        const Matrix< Scalar, Dynamic, 1 >  &qtb,
        Scalar delta,
        Matrix< Scalar, Dynamic, 1 >  &x)
{
    /* Local variables */
    int i, j, k;
    Scalar sum, temp, alpha, bnorm;
    Scalar gnorm, qnorm;
    Scalar sgnorm;

    /* Function Body */
    const Scalar epsmch = epsilon<Scalar>();
    const int n = diag.size();
    Matrix< Scalar, Dynamic, 1 >  wa1(n), wa2(n);
    assert(n==qtb.size());
    assert(n==x.size());

    for (k = 0; k < n; ++k) {
        j = n - k - 1;
        sum = 0.;
        for (i = j+1; i < n; ++i) {
            sum += qrfac(j,i) * x[i];
        }
        temp = qrfac(j,j);
        if (temp == 0.) {
            for (i = 0; i <= j; ++i)
                temp = std::max(temp,ei_abs(qrfac(i,j)));
            temp = epsmch * temp;
            if (temp == 0.)
                temp = epsmch;
        }
        x[j] = (qtb[j] - sum) / temp;
    }

    /* test whether the gauss-newton direction is acceptable. */

    wa1.fill(0.);
    wa2 = diag.cwiseProduct(x);
    qnorm = wa2.stableNorm();
    if (qnorm <= delta)
        return;

    /* the gauss-newton direction is not acceptable. */
    /* next, calculate the scaled gradient direction. */

    for (j = 0; j < n; ++j) {
        temp = qtb[j];
        for (i = j; i < n; ++i)
            wa1[i] += qrfac(j,i) * temp;
        wa1[j] /= diag[j];
    }

    /* calculate the norm of the scaled gradient and test for */
    /* the special case in which the scaled gradient is zero. */

    gnorm = wa1.stableNorm();
    sgnorm = 0.;
    alpha = delta / qnorm;
    if (gnorm == 0.)
        goto algo_end;

    /* calculate the point along the scaled gradient */
    /* at which the quadratic is minimized. */

    wa1.array() /= (diag*gnorm).array();
    for (j = 0; j < n; ++j) {
        sum = 0.;
        for (i = j; i < n; ++i) {
            sum += qrfac(j,i) * wa1[i];
        }
        wa2[j] = sum;
    }
    temp = wa2.stableNorm();
    sgnorm = gnorm / temp / temp;

    /* test whether the scaled gradient direction is acceptable. */

    alpha = 0.;
    if (sgnorm >= delta)
        goto algo_end;

    /* the scaled gradient direction is not acceptable. */
    /* finally, calculate the point along the dogleg */
    /* at which the quadratic is minimized. */

    bnorm = qtb.stableNorm();
    temp = bnorm / gnorm * (bnorm / qnorm) * (sgnorm / delta);
    /* Computing 2nd power */
    temp = temp - delta / qnorm * ei_abs2(sgnorm / delta) + ei_sqrt(ei_abs2(temp - delta / qnorm) + (1.-ei_abs2(delta / qnorm)) * (1.-ei_abs2(sgnorm / delta)));
    /* Computing 2nd power */
    alpha = delta / qnorm * (1. - ei_abs2(sgnorm / delta)) / temp;
algo_end:

    /* form appropriate convex combination of the gauss-newton */
    /* direction and the scaled gradient direction. */

    temp = (1.-alpha) * std::min(sgnorm,delta);
    x = temp * wa1 + alpha * x;
}

