
    template <typename Scalar>
void ei_rwupdt(int n, Scalar *r__, int ldr, const Scalar *w, Scalar *b, Scalar alpha)
{
    std::vector<PlanarRotation<Scalar> > givens(n);
    /* System generated locals */
    int r_dim1, r_offset;

    /* Local variables */
    Scalar temp, rowj;

    /* Parameter adjustments */
    --b;
    --w;
    r_dim1 = ldr;
    r_offset = 1 + r_dim1 * 1;
    r__ -= r_offset;

    /* Function Body */
    for (int j = 1; j <= n; ++j) {
        rowj = w[j];

        /* apply the previous transformations to */
        /* r(i,j), i=1,2,...,j-1, and to w(j). */
        if (j-1>=1)
            for (int i = 1; i <= j-1; ++i) {
                temp = givens[i-1].c() * r__[i + j * r_dim1] + givens[i-1].s() * rowj;
                rowj = -givens[i-1].s() * r__[i + j * r_dim1] + givens[i-1].c() * rowj;
                r__[i + j * r_dim1] = temp;
            }

        /* determine a givens rotation which eliminates w(j). */
        if (rowj != 0.) {
            givens[j-1].makeGivens(-r__[j + j * r_dim1], rowj);

            /* apply the current transformation to r(j,j), b(j), and alpha. */
            r__[j + j * r_dim1] = givens[j-1].c() * r__[j + j * r_dim1] + givens[j-1].s() * rowj;
            temp = givens[j-1].c() * b[j] + givens[j-1].s() * alpha;
            alpha = -givens[j-1].s() * b[j] + givens[j-1].c() * alpha;
            b[j] = temp;
        }
    }
    return;
}

