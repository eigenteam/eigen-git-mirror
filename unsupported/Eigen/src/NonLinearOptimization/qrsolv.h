
template <typename Scalar>
void ei_qrsolv(int n, Scalar *r__, int ldr, 
        const int *ipvt, const Scalar *diag, const Scalar *qtb, Scalar *x, 
        Scalar *sdiag)
{
    /* System generated locals */
    int r_dim1, r_offset;

    /* Local variables */
    int i, j, k, l;
    Scalar tan__, cos__, sin__, sum, temp, cotan;
    int nsing;
    Scalar qtbpj;
    Matrix< Scalar, Dynamic, 1 >  wa(n+1);

    /* Parameter adjustments */
    --sdiag;
    --x;
    --qtb;
    --diag;
    --ipvt;
    r_dim1 = ldr;
    r_offset = 1 + r_dim1 * 1;
    r__ -= r_offset;

    /* Function Body */

    /*     copy r and (q transpose)*b to preserve input and initialize s. */
    /*     in particular, save the diagonal elements of r in x. */

    for (j = 1; j <= n; ++j) {
        for (i = j; i <= n; ++i)
            r__[i + j * r_dim1] = r__[j + i * r_dim1];
        x[j] = r__[j + j * r_dim1];
        wa[j] = qtb[j];
    }

    /*     eliminate the diagonal matrix d using a givens rotation. */

    for (j = 1; j <= n; ++j) {

        /*        prepare the row of d to be eliminated, locating the */
        /*        diagonal element using p from the qr factorization. */

        l = ipvt[j];
        if (diag[l] == 0.)
            goto L90;
        for (k = j; k <= n; ++k)
            sdiag[k] = 0.;
        sdiag[j] = diag[l];

        /*        the transformations to eliminate the row of d */
        /*        modify only a single element of (q transpose)*b */
        /*        beyond the first n, which is initially zero. */

        qtbpj = 0.;
        for (k = j; k <= n; ++k) {
            /*           determine a givens rotation which eliminates the */
            /*           appropriate element in the current row of d. */
            if (sdiag[k] == 0.)
                continue;
            if ( ei_abs(r__[k + k * r_dim1]) < ei_abs(sdiag[k])) {
                cotan = r__[k + k * r_dim1] / sdiag[k];
                /* Computing 2nd power */
                sin__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
                cos__ = sin__ * cotan;
            } else {
                tan__ = sdiag[k] / r__[k + k * r_dim1];
                /* Computing 2nd power */
                cos__ = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
                sin__ = cos__ * tan__;
            }

            /*           compute the modified diagonal element of r and */
            /*           the modified element of ((q transpose)*b,0). */

            r__[k + k * r_dim1] = cos__ * r__[k + k * r_dim1] + sin__ * sdiag[
                k];
            temp = cos__ * wa[k] + sin__ * qtbpj;
            qtbpj = -sin__ * wa[k] + cos__ * qtbpj;
            wa[k] = temp;

            /*           accumulate the tranformation in the row of s. */
            for (i = k+1; i <= n; ++i) {
                temp = cos__ * r__[i + k * r_dim1] + sin__ * sdiag[i];
                sdiag[i] = -sin__ * r__[i + k * r_dim1] + cos__ * sdiag[i];
                r__[i + k * r_dim1] = temp;
            }
        }
L90:

        /*        store the diagonal element of s and restore */
        /*        the corresponding diagonal element of r. */

        sdiag[j] = r__[j + j * r_dim1];
        r__[j + j * r_dim1] = x[j];
    }

    /*     solve the triangular system for z. if the system is */
    /*     singular, then obtain a least squares solution. */

    nsing = n;
    for (j = 1; j <= n; ++j) {
        if (sdiag[j] == 0. && nsing == n) nsing = j - 1;
        if (nsing < n) wa[j] = 0.;
    }
    if (nsing >= 1)
        for (k = 1; k <= nsing; ++k) {
            j = nsing - k + 1;
            sum = 0.;
            if (nsing>j)
                for (i = j+1; i <= nsing; ++i)
                    sum += r__[i + j * r_dim1] * wa[i];
            wa[j] = (wa[j] - sum) / sdiag[j];
        }

    /*     permute the components of z back to components of x. */
    for (j = 1; j <= n; ++j) {
        l = ipvt[j];
        x[l] = wa[j];
    }
}

