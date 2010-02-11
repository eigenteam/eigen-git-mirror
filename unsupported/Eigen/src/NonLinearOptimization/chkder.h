
#define chkder_log10e 0.43429448190325182765
#define chkder_factor 100.

template<typename Scalar>
void ei_chkder(
        const Matrix< Scalar, Dynamic, 1 >  &x,
        const Matrix< Scalar, Dynamic, 1 >  &fvec,
        const Matrix< Scalar, Dynamic, Dynamic > &fjac,
        Matrix< Scalar, Dynamic, 1 >  &xp,
        const Matrix< Scalar, Dynamic, 1 >  &fvecp,
        int mode,
        Matrix< Scalar, Dynamic, 1 >  &err
        )
{
    const Scalar eps = ei_sqrt(NumTraits<Scalar>::epsilon());
    const Scalar epsf = chkder_factor * NumTraits<Scalar>::epsilon();
    const Scalar epslog = chkder_log10e * ei_log(eps);
    Scalar temp;

    const int m = fvec.size(), n = x.size();

    if (mode != 2) {
        /* mode = 1. */
        xp.resize(n);
        for (int j = 0; j < n; ++j) {
            temp = eps * ei_abs(x[j]);
            if (temp == 0.)
                temp = eps;
            xp[j] = x[j] + temp;
        }
    }
    else {
        /* mode = 2. */
        err.setZero(m); 
        for (int j = 0; j < n; ++j) {
            temp = ei_abs(x[j]);
            if (temp == 0.)
                temp = 1.;
            err += temp * fjac.col(j);
        }
        for (int i = 0; i < m; ++i) {
            temp = 1.;
            if (fvec[i] != 0. && fvecp[i] != 0. && ei_abs(fvecp[i] - fvec[i]) >= epsf * ei_abs(fvec[i]))
                temp = eps * ei_abs((fvecp[i] - fvec[i]) / eps - err[i]) / (ei_abs(fvec[i]) + ei_abs(fvecp[i]));
            err[i] = 1.;
            if (temp > NumTraits<Scalar>::epsilon() && temp < eps)
                err[i] = (chkder_log10e * ei_log(temp) - epslog) / epslog;
            if (temp >= eps)
                err[i] = 0.;
        }
    }
}

