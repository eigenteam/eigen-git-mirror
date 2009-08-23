
#define chkder_log10e 0.43429448190325182765
#define chkder_factor 100.

template<typename Scalar>
void ei_chkder(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        Matrix< Scalar, Dynamic, 1 >  &xp,
        Matrix< Scalar, Dynamic, 1 >  &fvecp,
        int mode,
        Matrix< Scalar, Dynamic, 1 >  &err
        )
{
    const Scalar eps = ei_sqrt(epsilon<Scalar>());
    const Scalar epsf = chkder_factor * epsilon<Scalar>();
    const Scalar epslog = chkder_log10e * ei_log(eps);
    Scalar temp;
    int i,j;

    const int m = fvec.size(), n = x.size();

    if (mode != 2) {
        xp.resize(m);
        /*        mode = 1. */
        for (j = 0; j < n; ++j) {
            temp = eps * ei_abs(x[j]);
            if (temp == 0.)
                temp = eps;
            xp[j] = x[j] + temp;
        }
    }
    else {
        /*        mode = 2. */
        err.setZero(m); 
        for (j = 0; j < n; ++j) {
            temp = ei_abs(x[j]);
            if (temp == 0.)
                temp = 1.;
            err += temp * fjac.col(j);
        }
        for (i = 0; i < m; ++i) {
            temp = 1.;
            if (fvec[i] != 0. && fvecp[i] != 0. && ei_abs(fvecp[i] - fvec[i]) >= epsf * ei_abs(fvec[i]))
                temp = eps * ei_abs((fvecp[i] - fvec[i]) / eps - err[i]) / (ei_abs(fvec[i]) + ei_abs(fvecp[i]));
            err[i] = 1.;
            if (temp > epsilon<Scalar>() && temp < eps)
                err[i] = (chkder_log10e * ei_log(temp) - epslog) / epslog;
            if (temp >= eps)
                err[i] = 0.;
        }
    }
} /* chkder_ */

