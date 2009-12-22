
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
    nm1 = n - 1;
    if (nm1 < 1)
        return;

    /*     apply the first set of givens rotations to a. */
    for (nmj = 1; nmj <= nm1; ++nmj) {
        j = n - nmj;
        if (ei_abs(v[j]) > 1.) {
            cos__ = 1. / v[j];
            sin__ = ei_sqrt(1. - ei_abs2(cos__));
        } else  {
            sin__ = v[j];
            cos__ = ei_sqrt(1. - ei_abs2(sin__));
        }
        for (i = 1; i <= m; ++i) {
            temp = cos__ * a[i + j * a_dim1] - sin__ * a[i + n * a_dim1];
            a[i + n * a_dim1] = sin__ * a[i + j * a_dim1] + cos__ * a[
                i + n * a_dim1];
            a[i + j * a_dim1] = temp;
        }
    }
    /*     apply the second set of givens rotations to a. */
    for (j = 1; j <= nm1; ++j) {
        if (ei_abs(w[j]) > 1.) {
            cos__ = 1. / w[j];
            sin__ = ei_sqrt(1. - ei_abs2(cos__));
        } else  {
            sin__ = w[j];
            cos__ = ei_sqrt(1. - ei_abs2(sin__));
        }
        for (i = 1; i <= m; ++i) {
            temp = cos__ * a[i + j * a_dim1] + sin__ * a[i + n * a_dim1];
            a[i + n * a_dim1] = -sin__ * a[i + j * a_dim1] + cos__ * a[
                i + n * a_dim1];
            a[i + j * a_dim1] = temp;
        }
    }
    return;
} /* r1mpyq_ */

