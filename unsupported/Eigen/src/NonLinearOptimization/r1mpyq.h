
// TODO : move this to GivensQR once there's such a thing in Eigen

template <typename Scalar>
void ei_r1mpyq(int m, int n, Scalar *a, const Scalar *v, const Scalar *w)
{
    /* Local variables */
    int i, j;
    Scalar cos__=0., sin__=0., temp;

    /* Function Body */
    if (n<=1)
        return;

    /*     apply the first set of givens rotations to a. */
    for (j = n-2; j>=0; --j) {
        if (ei_abs(v[j]) > 1.) {
            cos__ = 1. / v[j];
            sin__ = ei_sqrt(1. - ei_abs2(cos__));
        } else  {
            sin__ = v[j];
            cos__ = ei_sqrt(1. - ei_abs2(sin__));
        }
        for (i = 0; i<m; ++i) {
            temp = cos__ * a[i+m*j] - sin__ * a[i+m*(n-1)];
            a[i+m*(n-1)] = sin__ * a[i+m*j] + cos__ * a[i+m*(n-1)];
            a[i+m*j] = temp;
        }
    }
    /*     apply the second set of givens rotations to a. */
    for (j = 0; j<n-1; ++j) {
        if (ei_abs(w[j]) > 1.) {
            cos__ = 1. / w[j];
            sin__ = ei_sqrt(1. - ei_abs2(cos__));
        } else  {
            sin__ = w[j];
            cos__ = ei_sqrt(1. - ei_abs2(sin__));
        }
        for (i = 0; i<m; ++i) {
            temp = cos__ * a[i+m*j] + sin__ * a[i+m*(n-1)];
            a[i+m*(n-1)] = -sin__ * a[i+m*j] + cos__ * a[i+m*(n-1)];
            a[i+m*j] = temp;
        }
    }
    return;
}

