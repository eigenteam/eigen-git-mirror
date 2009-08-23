
template<typename Functor, typename Scalar>
int ei_fdjac2(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        Scalar epsfcn,
        Matrix< Scalar, Dynamic, 1 >  &wa)
{
    /* Local variables */
    Scalar h, temp;
    int iflag;

    /* Function Body */
    const Scalar epsmch = epsilon<Scalar>();
    const int n = x.size();
    const Scalar eps = ei_sqrt((std::max(epsfcn,epsmch)));

    for (int j = 0; j < n; ++j) {
        temp = x[j];
        h = eps * ei_abs(temp);
        if (h == 0.) {
            h = eps;
        }
        x[j] = temp + h;
        iflag = Functor::f(x, wa);
        if (iflag < 0)
            return iflag;
        x[j] = temp;
        fjac.col(j) = (wa-fvec)/h;
    }
    return iflag;
}

