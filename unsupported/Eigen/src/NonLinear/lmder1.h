
template<typename Functor, typename Scalar>
int ei_lmder1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        VectorXi &ipvt,
        Scalar tol = ei_sqrt(epsilon<Scalar>())
        )
{
    const int n = x.size(), m=fvec.size();
    int info, nfev=0, njev=0;
    Matrix< Scalar, Dynamic, Dynamic > fjac(m, n);
    Matrix< Scalar, Dynamic, 1> diag, qtf;

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.) {
        printf("ei_lmder1 bad args : m,n,tol,...");
        return 0;
    }

    ipvt.resize(n);
    info = ei_lmder<Functor,Scalar>(
        x, fvec,
        nfev, njev,
        fjac, ipvt, qtf, diag,
        1,
        100.,
        (n+1)*100,
        tol, tol, Scalar(0.)
    );
    return (info==8)?4:info;
}

