
template<typename Functor, typename Scalar>
int ei_hybrd1(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Scalar tol = ei_sqrt(epsilon<Scalar>())
        )
{
    const int n = x.size();
    int info, nfev=0;
    Matrix< Scalar, Dynamic, Dynamic > fjac;
    Matrix< Scalar, Dynamic, 1> R, qtf, diag;

    /* check the input parameters for errors. */
    if (n <= 0 || tol < 0.) {
        printf("ei_hybrd1 bad args : n,tol,...");
        return 0;
    }

    diag.setConstant(n, 1.);
    info = ei_hybrd<Functor,Scalar>(
        x, fvec,
        nfev,
        fjac,
        R, qtf, diag,
        2,
        -1, -1,
        (n+1)*200,
        100.,
        tol, Scalar(0.)
    );
    return (info==5)?4:info;
}

