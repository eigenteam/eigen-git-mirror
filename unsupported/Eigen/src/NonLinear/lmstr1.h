
template<typename Functor, typename Scalar>
int ei_lmstr1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        VectorXi &ipvt,
        Scalar tol = ei_sqrt(epsilon<Scalar>())
        )
{
    const int n = x.size(), m=fvec.size();
    int info, nfev=0, njev=0;
    Matrix< Scalar, Dynamic, Dynamic > fjac(m, n);
    Matrix< Scalar, Dynamic, 1> diag;

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.) {
        printf("ei_lmstr1 bad args : m,n,tol,...");
        return 0;
    }

    ipvt.resize(n);
    info = ei_lmstr<Functor,Scalar>(
        x, fvec,
        nfev, njev,
        fjac, ipvt, diag,
        1,
        100.,
        (n+1)*100,
        tol, tol, Scalar(0.)
    );
    return (info==8)?4:info;
}

