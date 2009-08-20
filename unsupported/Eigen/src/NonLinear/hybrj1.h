
template<typename Functor, typename Scalar>
int ei_hybrj1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        Scalar tol = ei_sqrt(epsilon<Scalar>())
        )
{
    const int n = x.size();
    int info, nfev, njev;
    Matrix< Scalar, Dynamic, 1> R, qtf, diag;

    /* check the input parameters for errors. */
    if (n <= 0 || tol < 0.) {
        printf("ei_hybrd1 bad args : n,tol,...");
        return 0;
    }

    diag.setConstant(n, 1.);
    info = ei_hybrj<Functor,Scalar>(
        x, fvec,
        nfev, njev,
        fjac,
        R, qtf, diag,
        2,
        (n+1)*100,
        100.,
        tol
    );
    return (info==5)?4:info;
}

