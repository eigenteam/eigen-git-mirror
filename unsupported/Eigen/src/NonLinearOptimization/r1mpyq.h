
template <typename Scalar>
void ei_r1mpyq(int m, int n, Scalar *a, int
        lda, const Scalar *v, const Scalar *w)
{
    /* System generated locals */
    int a_dim1, a_offset;

    /* Local variables */
    int i, j, nm1, nmj;
    Scalar cos__=0., sin__=0., temp;

    /* Parameter adjustments */
    --w;
    --v;
    a_dim1 = lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;

    /* Function Body */

    /*     apply the first set of givens rotations to a. */

    nm1 = n - 1;
    if (nm1 < 1) {
        /* goto L50; */
        return;
    }
    for (nmj = 1; nmj <= nm1; ++nmj) {
        j = n - nmj;
        if (ei_abs(v[j]) > 1.) {
            cos__ = 1. / v[j];
        }
        if (ei_abs(v[j]) > 1.) {
            /* Computing 2nd power */
            sin__ = ei_sqrt(1. - ei_abs2(cos__));
        }
        if (ei_abs(v[j]) <= 1.) {
            sin__ = v[j];
        }
        if (ei_abs(v[j]) <= 1.) {
            /* Computing 2nd power */
            cos__ = ei_sqrt(1. - ei_abs2(sin__));
        }
        for (i = 1; i <= m; ++i) {
            temp = cos__ * a[i + j * a_dim1] - sin__ * a[i + n * a_dim1];
            a[i + n * a_dim1] = sin__ * a[i + j * a_dim1] + cos__ * a[
                i + n * a_dim1];
            a[i + j * a_dim1] = temp;
            /* L10: */
        }
        /* L20: */
    }

    /*     apply the second set of givens rotations to a. */

    for (j = 1; j <= nm1; ++j) {
        if (ei_abs(w[j]) > 1.) {
            cos__ = 1. / w[j];
        }
        if (ei_abs(w[j]) > 1.) {
            /* Computing 2nd power */
            sin__ = ei_sqrt(1. - ei_abs2(cos__));
        }
        if (ei_abs(w[j]) <= 1.) {
            sin__ = w[j];
        }
        if (ei_abs(w[j]) <= 1.) {
            /* Computing 2nd power */
            cos__ = ei_sqrt(1. - ei_abs2(sin__));
        }
        for (i = 1; i <= m; ++i) {
            temp = cos__ * a[i + j * a_dim1] + sin__ * a[i + n * a_dim1];
            a[i + n * a_dim1] = -sin__ * a[i + j * a_dim1] + cos__ * a[
                i + n * a_dim1];
            a[i + j * a_dim1] = temp;
            /* L30: */
        }
        /* L40: */
    }
    /* L50: */
    return;

    /*     last card of subroutine r1mpyq. */

} /* r1mpyq_ */

