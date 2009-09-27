
template<typename FunctorType, typename Scalar>
int ei_fdjac1(
        const FunctorType &Functor,
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        int ml, int mu,
        Scalar epsfcn)
{
    /* Local variables */
    Scalar h;
    int i, j, k;
    Scalar eps, temp;
    int msum;
    int iflag = 0;

    /* Function Body */
    const Scalar epsmch = epsilon<Scalar>();
    const int n = x.size();
    assert(fvec.size()==n);
    Matrix< Scalar, Dynamic, 1 >  wa1(n);
    Matrix< Scalar, Dynamic, 1 >  wa2(n);

    eps = ei_sqrt((std::max(epsfcn,epsmch)));
    msum = ml + mu + 1;
    if (msum >= n) {
        /* computation of dense approximate jacobian. */
        for (j = 0; j < n; ++j) {
            temp = x[j];
            h = eps * ei_abs(temp);
            if (h == 0.)
                h = eps;
            x[j] = temp + h;
            iflag = Functor(x, wa1);
            if (iflag < 0)
                return iflag;
            x[j] = temp;
            fjac.col(j) = (wa1-fvec)/h;
        }

    }else {
        /* computation of banded approximate jacobian. */
        for (k = 0; k < msum; ++k) {
            for (j = k; msum< 0 ? j > n: j < n; j += msum) {
                wa2[j] = x[j];
                h = eps * ei_abs(wa2[j]);
                if (h == 0.) h = eps;
                x[j] = wa2[j] + h;
            }
            iflag = Functor(x, wa1);
            if (iflag < 0) {
                return iflag;
            }
            for (j = k; msum< 0 ? j > n: j < n; j += msum) {
                x[j] = wa2[j];
                h = eps * ei_abs(wa2[j]);
                if (h == 0.) h = eps;
                for (i = 0; i < n; ++i) {
                    fjac(i,j) = 0.;
                    if (i >= j - mu && i <= j + ml) {
                        fjac(i,j) = (wa1[i] - fvec[i]) / h;
                    }
                }
            }
        }
    }
    return iflag;
} /* fdjac1_ */

