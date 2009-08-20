
#define chkder_log10e 0.43429448190325182765
#define chkder_factor 100.

/* Table of constant values */

    template<typename Scalar>
void chkder_template(int m, int n, const Scalar *x, 
        Scalar *fvec, Scalar *fjac, int ldfjac, Scalar *xp, 
        Scalar *fvecp, int mode, Scalar *err)
{
    /* System generated locals */
    int fjac_offset;

    /* Local variables */
    int i__, j;
    Scalar temp;

    /* Parameter adjustments */
    --err;
    --fvecp;
    --fvec;
    --xp;
    --x;
    fjac_offset = 1 + ldfjac;
    fjac -= fjac_offset;

    /* Function Body */

    const Scalar eps = ei_sqrt(epsilon<Scalar>());

    if (mode != 2) {
        /*        mode = 1. */
        for (j = 1; j <= n; ++j) {
            temp = eps * fabs(x[j]);
            if (temp == 0.) {
                temp = eps;
            }
            xp[j] = x[j] + temp;
        }
        return;
    }

    /*        mode = 2. */
    const Scalar epsf = chkder_factor * epsilon<Scalar>();
    const Scalar epslog = chkder_log10e * log(eps);
    for (i__ = 1; i__ <= m; ++i__) {
        err[i__] = 0.;
    }
    for (j = 1; j <= n; ++j) {
        temp = fabs(x[j]);
        if (temp == 0.) {
            temp = 1.;
        }
        for (i__ = 1; i__ <= m; ++i__) {
            err[i__] += temp * fjac[i__ + j * ldfjac];
        }
    }
    for (i__ = 1; i__ <= m; ++i__) {
        temp = 1.;
        if (fvec[i__] != 0. && fvecp[i__] != 0. && fabs(fvecp[i__] - 
                    fvec[i__]) >= epsf * fabs(fvec[i__]))
        {
            temp = eps * fabs((fvecp[i__] - fvec[i__]) / eps - err[i__]) 
                / (fabs(fvec[i__]) +
                        fabs(fvecp[i__]));
        }
        err[i__] = 1.;
        if (temp > epsilon<Scalar>() && temp < eps) {
            err[i__] = (chkder_log10e * log(temp) - epslog) / epslog;
        }
        if (temp >= eps) {
            err[i__] = 0.;
        }
    }
} /* chkder_ */

