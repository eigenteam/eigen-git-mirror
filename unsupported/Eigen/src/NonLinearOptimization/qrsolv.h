
#if 0
        int n, Scalar *r__, int ldr, 
        const int *ipvt, const Scalar *diag, const Scalar *qtb, Scalar *x, 
        Scalar *sdiag)
#endif


template <typename Scalar>
void ei_qrsolv(
        Matrix< Scalar, Dynamic, Dynamic > &r,
        VectorXi &ipvt, // TODO : const once ipvt mess fixed
        const Matrix< Scalar, Dynamic, 1 >  &diag,
        const Matrix< Scalar, Dynamic, 1 >  &qtb,
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &sdiag)

{
    /* Local variables */
    int i, j, k, l;
    Scalar tan__, cos__, sin__, sum, temp, cotan;
    int nsing;
    Scalar qtbpj;
    int n = r.cols();
    Matrix< Scalar, Dynamic, 1 >  wa(n);

    /* Function Body */

    /*     copy r and (q transpose)*b to preserve input and initialize s. */
    /*     in particular, save the diagonal elements of r in x. */

    for (j = 0; j < n; ++j) {
        for (i = j; i < n; ++i)
            r(i,j) = r(j,i);
        x[j] = r(j,j);
        wa[j] = qtb[j];
    }

    /*     eliminate the diagonal matrix d using a givens rotation. */
    for (j = 0; j < n; ++j) {

        /*        prepare the row of d to be eliminated, locating the */
        /*        diagonal element using p from the qr factorization. */

        l = ipvt[j];
        if (diag[l] == 0.)
            goto L90;
        for (k = j; k < n; ++k)
            sdiag[k] = 0.;
        sdiag[j] = diag[l];

        /*        the transformations to eliminate the row of d */
        /*        modify only a single element of (q transpose)*b */
        /*        beyond the first n, which is initially zero. */

        qtbpj = 0.;
        for (k = j; k < n; ++k) {
            /*           determine a givens rotation which eliminates the */
            /*           appropriate element in the current row of d. */
            if (sdiag[k] == 0.)
                continue;
            if ( ei_abs(r(k,k)) < ei_abs(sdiag[k])) {
                cotan = r(k,k) / sdiag[k];
                /* Computing 2nd power */
                sin__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
                cos__ = sin__ * cotan;
            } else {
                tan__ = sdiag[k] / r(k,k);
                /* Computing 2nd power */
                cos__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
                sin__ = cos__ * tan__;
            }

            /*           compute the modified diagonal element of r and */
            /*           the modified element of ((q transpose)*b,0). */

            r(k,k) = cos__ * r(k,k) + sin__ * sdiag[k];
            temp = cos__ * wa[k] + sin__ * qtbpj;
            qtbpj = -sin__ * wa[k] + cos__ * qtbpj;
            wa[k] = temp;

            /*           accumulate the tranformation in the row of s. */
            for (i = k+1; i<n; ++i) {
                temp = cos__ * r(i,k) + sin__ * sdiag[i];
                sdiag[i] = -sin__ * r(i,k) + cos__ * sdiag[i];
                r(i,k) = temp;
            }
        }
L90:

        /*        store the diagonal element of s and restore */
        /*        the corresponding diagonal element of r. */

        sdiag[j] = r(j,j);
        r(j,j) = x[j];
    }

    /*     solve the triangular system for z. if the system is */
    /*     singular, then obtain a least squares solution. */

    nsing = n-1;
    for (j = 0; j < n; ++j) {
        if (sdiag[j] == 0. && nsing == n-1) nsing = j - 1;
        if (nsing < n-1) wa[j] = 0.;
    }
    for (k = 0; k <= nsing; ++k) {
        j = nsing - k;
        sum = 0.;
        for (i = j+1; i <= nsing; ++i)
            sum += r(i,j) * wa[i];
        wa[j] = (wa[j] - sum) / sdiag[j];
    }

    /*     permute the components of z back to components of x. */
    for (j = 0; j < n; ++j) {
        l = ipvt[j];
        x[l] = wa[j];
    }
}

