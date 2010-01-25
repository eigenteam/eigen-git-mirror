
template <typename Scalar>
void ei_qrsolv(
        Matrix< Scalar, Dynamic, Dynamic > &r,
        const VectorXi &ipvt,
        const Matrix< Scalar, Dynamic, 1 >  &diag,
        const Matrix< Scalar, Dynamic, 1 >  &qtb,
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &sdiag)

{
    /* Local variables */
    int i, j, k, l;
    Scalar sum, temp;
    int n = r.cols();
    Matrix< Scalar, Dynamic, 1 >  wa(n);

    /* Function Body */

    /*     copy r and (q transpose)*b to preserve input and initialize s. */
    /*     in particular, save the diagonal elements of r in x. */

    x = r.diagonal();
    wa = qtb;

    for (j = 0; j < n; ++j)
        for (i = j+1; i < n; ++i)
            r(i,j) = r(j,i);

    /*     eliminate the diagonal matrix d using a givens rotation. */
    for (j = 0; j < n; ++j) {

        /*        prepare the row of d to be eliminated, locating the */
        /*        diagonal element using p from the qr factorization. */

        l = ipvt[j];
        if (diag[l] == 0.)
            break;
        sdiag.segment(j,n-j).setZero();
        sdiag[j] = diag[l];

        /*        the transformations to eliminate the row of d */
        /*        modify only a single element of (q transpose)*b */
        /*        beyond the first n, which is initially zero. */

        Scalar qtbpj = 0.;
        for (k = j; k < n; ++k) {
            /*           determine a givens rotation which eliminates the */
            /*           appropriate element in the current row of d. */
            PlanarRotation<Scalar> givens;
            givens.makeGivens(-r(k,k), sdiag[k]);

            /*           compute the modified diagonal element of r and */
            /*           the modified element of ((q transpose)*b,0). */

            r(k,k) = givens.c() * r(k,k) + givens.s() * sdiag[k];
            temp = givens.c() * wa[k] + givens.s() * qtbpj;
            qtbpj = -givens.s() * wa[k] + givens.c() * qtbpj;
            wa[k] = temp;

            /*           accumulate the tranformation in the row of s. */
            for (i = k+1; i<n; ++i) {
                temp = givens.c() * r(i,k) + givens.s() * sdiag[i];
                sdiag[i] = -givens.s() * r(i,k) + givens.c() * sdiag[i];
                r(i,k) = temp;
            }
        }
    }

    // restore
    sdiag = r.diagonal();
    r.diagonal() = x;

    /*     solve the triangular system for z. if the system is */
    /*     singular, then obtain a least squares solution. */

    int nsing;
    for (nsing=0; nsing<n && sdiag[nsing]!=0; nsing++);
    wa.segment(nsing,n-nsing).setZero();
    nsing--; // nsing is the last nonsingular index

    for (j = nsing; j>=0; j--) {
        sum = 0.;
        for (i = j+1; i <= nsing; ++i)
            sum += r(i,j) * wa[i];
        wa[j] = (wa[j] - sum) / sdiag[j];
    }

    /*     permute the components of z back to components of x. */
    for (j = 0; j < n; ++j) x[ipvt[j]] = wa[j];
}

