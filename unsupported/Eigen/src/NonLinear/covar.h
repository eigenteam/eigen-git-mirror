
    template <typename Scalar>
void ei_covar(int n, Scalar *r__, int ldr, 
        const int *ipvt, Scalar tol, Scalar *wa)
{
    /* System generated locals */
    int r_dim1, r_offset;

    /* Local variables */
    int i, j, k, l, ii, jj, km1;
    int sing;
    Scalar temp, tolr;

    /* Parameter adjustments */
    --wa;
    --ipvt;
    tolr = tol * ei_abs(r__[0]);
    r_dim1 = ldr;
    r_offset = 1 + r_dim1;
    r__ -= r_offset;

    /* Function Body */

    /*     form the inverse of r in the full upper triangle of r. */

    l = 0;
    for (k = 1; k <= n; ++k) {
        if (ei_abs(r__[k + k * r_dim1]) > tolr) {
            r__[k + k * r_dim1] = 1. / r__[k + k * r_dim1];
            km1 = k - 1;
            if (km1 >= 1)
                for (j = 1; j <= km1; ++j) {
                    temp = r__[k + k * r_dim1] * r__[j + k * r_dim1];
                    r__[j + k * r_dim1] = 0.;
                    for (i = 1; i <= j; ++i) {
                        r__[i + k * r_dim1] -= temp * r__[i + j * r_dim1];
                    }
                }
            l = k;
        }
    }

    /*     form the full upper triangle of the inverse of (r transpose)*r */
    /*     in the full upper triangle of r. */

    if (l >= 1)
        for (k = 1; k <= l; ++k) {
            km1 = k - 1;
            if (km1 >= 1)
                for (j = 1; j <= km1; ++j) {
                    temp = r__[j + k * r_dim1];
                    for (i = 1; i <= j; ++i)
                        r__[i + j * r_dim1] += temp * r__[i + k * r_dim1];
                }
            temp = r__[k + k * r_dim1];
            for (i = 1; i <= k; ++i)
                r__[i + k * r_dim1] = temp * r__[i + k * r_dim1];
        }

    /*     form the full lower triangle of the covariance matrix */
    /*     in the strict lower triangle of r and in wa. */

    for (j = 1; j <= n; ++j) {
        jj = ipvt[j];
        sing = j > l;
        for (i = 1; i <= j; ++i) {
            if (sing)
                r__[i + j * r_dim1] = 0.;
            ii = ipvt[i];
            if (ii > jj)
                r__[ii + jj * r_dim1] = r__[i + j * r_dim1];
            if (ii < jj)
                r__[jj + ii * r_dim1] = r__[i + j * r_dim1];
        }
        wa[jj] = r__[j + j * r_dim1];
    }

    /*     symmetrize the covariance matrix in r. */

    for (j = 1; j <= n; ++j) {
        for (i = 1; i <= j; ++i)
            r__[i + j * r_dim1] = r__[j + i * r_dim1];
        r__[j + j * r_dim1] = wa[j];
    }
}

