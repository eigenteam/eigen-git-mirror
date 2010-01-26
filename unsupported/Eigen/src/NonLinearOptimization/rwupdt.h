
template <typename Scalar>
void ei_rwupdt(int n, Scalar *r__, int ldr, 
	const Scalar *w, Scalar *b, Scalar *alpha, Scalar *cos__, 
	Scalar *sin__)
{
    /* System generated locals */
    int r_dim1, r_offset;

    /* Local variables */
    Scalar tan__, temp, rowj, cotan;

    /* Parameter adjustments */
    --sin__;
    --cos__;
    --b;
    --w;
    r_dim1 = ldr;
    r_offset = 1 + r_dim1 * 1;
    r__ -= r_offset;

    /* Function Body */
    for (int j = 1; j <= n; ++j) {
        rowj = w[j];

        /*        apply the previous transformations to */
        /*        r(i,j), i=1,2,...,j-1, and to w(j). */
        if (j-1>=1)
            for (int i = 1; i <= j-1; ++i) {
                temp = cos__[i] * r__[i + j * r_dim1] + sin__[i] * rowj;
                rowj = -sin__[i] * r__[i + j * r_dim1] + cos__[i] * rowj;
                r__[i + j * r_dim1] = temp;
            }

        /*        determine a givens rotation which eliminates w(j). */
        cos__[j] = 1.;
        sin__[j] = 0.;
        if (rowj != 0.) {
            if (ei_abs(r__[j + j * r_dim1]) < ei_abs(rowj)) {
                cotan = r__[j + j * r_dim1] / rowj;
                sin__[j] = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(cotan));
                cos__[j] = sin__[j] * cotan;
            }
            else {
                tan__ = rowj / r__[j + j * r_dim1];
                cos__[j] = Scalar(.5) / ei_sqrt(Scalar(0.25) + Scalar(0.25) * ei_abs2(tan__));
                sin__[j] = cos__[j] * tan__;
            }

            /*        apply the current transformation to r(j,j), b(j), and alpha. */
            r__[j + j * r_dim1] = cos__[j] * r__[j + j * r_dim1] + sin__[j] * rowj;
            temp = cos__[j] * b[j] + sin__[j] * *alpha;
            *alpha = -sin__[j] * b[j] + cos__[j] * *alpha;
            b[j] = temp;
        }
    }
    return;
}

