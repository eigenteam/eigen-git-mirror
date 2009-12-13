// -*- coding: utf-8
// vim: set fileencoding=utf-8

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_LEVENBERGMARQUARDT__H
#define EIGEN_LEVENBERGMARQUARDT__H

/**
  * \ingroup NonLinearOptimization_Module
  * \brief Performs non linear optimization over a non-linear function,
  * using a variant of the Levenberg Marquardt algorithm.
  *
  * Check wikipedia for more information.
  * http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
  */
template<typename FunctorType, typename Scalar=double>
class LevenbergMarquardt 
{
public:
    LevenbergMarquardt(FunctorType &_functor)
        : functor(_functor) { nfev = njev = iter = 0;  fnorm=gnorm = 0.; }

    enum Status {
        Running = -1,
        ImproperInputParameters = 0,
        RelativeReductionTooSmall = 1,
        RelativeErrorTooSmall = 2,
        RelativeErrorAndReductionTooSmall = 3,
        CosinusTooSmall = 4,
        TooManyFunctionEvaluation = 5, 
        FtolTooSmall = 6,
        XtolTooSmall = 7,
        GtolTooSmall = 8,
        UserAsked = 9
    };

    struct Parameters {
        Parameters()
            : factor(Scalar(100.))
            , maxfev(400)
            , ftol(ei_sqrt(epsilon<Scalar>()))
            , xtol(ei_sqrt(epsilon<Scalar>()))
            , gtol(Scalar(0.))
            , epsfcn(Scalar(0.)) {}
        Scalar factor;
        int maxfev;   // maximum number of function evaluation
        Scalar ftol;
        Scalar xtol;
        Scalar gtol;
        Scalar epsfcn;
    };

    typedef Matrix< Scalar, Dynamic, 1 > FVectorType;
    typedef Matrix< Scalar, Dynamic, Dynamic > JacobianType;

    Status lmder1(
            FVectorType &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status minimize(
            FVectorType &x,
            const int mode=1
            );
    Status minimizeInit(
            FVectorType &x,
            const int mode=1
            );
    Status minimizeOneStep(
            FVectorType &x,
            const int mode=1
            );

    static Status lmdif1(
            FunctorType &functor,
            FVectorType &x,
            int *nfev,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status lmstr1(
            FVectorType  &x,
            const Scalar tol = ei_sqrt(epsilon<Scalar>())
            );

    Status minimizeOptimumStorage(
            FVectorType  &x,
            const int mode=1
            );
    Status minimizeOptimumStorageInit(
            FVectorType  &x,
            const int mode=1
            );
    Status minimizeOptimumStorageOneStep(
            FVectorType  &x,
            const int mode=1
            );

    void resetParameters(void) { parameters = Parameters(); }

    Parameters parameters;
    FVectorType  fvec, qtf, diag;
    JacobianType fjac;
    VectorXi ipvt;
    int nfev;
    int njev;
    int iter;
    Scalar fnorm, gnorm;
private:
    FunctorType &functor;
    int n;
    int m;
    FVectorType wa1, wa2, wa3, wa4;

    Scalar par, sum;
    Scalar temp, temp1, temp2;
    Scalar delta;
    Scalar ratio;
    Scalar pnorm, xnorm, fnorm1, actred, dirder, prered;
};

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::lmder1(
        FVectorType  &x,
        const Scalar tol
        )
{
    n = x.size();
    m = functor.values();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return ImproperInputParameters;

    resetParameters();
    parameters.ftol = tol;
    parameters.xtol = tol;
    parameters.maxfev = 100*(n+1);

    return minimize(x);
}


template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimize(
        FVectorType  &x,
        const int mode
        )
{
    Status status = minimizeInit(x, mode);
    while (status==Running)
        status = minimizeOneStep(x, mode);
    return status;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeInit(
        FVectorType  &x,
        const int mode
        )
{
    n = x.size();
    m = functor.values();

    wa1.resize(n); wa2.resize(n); wa3.resize(n);
    wa4.resize(m);
    fvec.resize(m);
    ipvt.resize(n);
    fjac.resize(m, n);
    if (mode != 2)
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");
    qtf.resize(n);

    /* Function Body */
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || parameters.ftol < 0. || parameters.xtol < 0. || parameters.gtol < 0. || parameters.maxfev <= 0 || parameters.factor <= 0.)
        return ImproperInputParameters;

    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor(x, fvec) < 0)
        return UserAsked;
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOneStep(
        FVectorType  &x,
        const int mode
        )
{
    int i, j, l;

    /* calculate the jacobian matrix. */

    int df_ret = functor.df(x, fjac);
    if (df_ret<0)
        return UserAsked;
    if (df_ret>0)
        // numerical diff, we evaluated the function df_ret times
        nfev += df_ret;
    else njev++;

    /* compute the qr factorization of the jacobian. */

    wa2 = fjac.colwise().blueNorm();
    ei_qrfac<Scalar>(m, n, fjac.data(), fjac.rows(), true, ipvt.data(), wa1.data());
    ipvt.cwise()-=1; // qrfac() creates ipvt with fortran convention (1->n), convert it to c (0->n-1)

    /* on the first iteration and if mode is 1, scale according */
    /* to the norms of the columns of the initial jacobian. */

    if (iter == 1) {
        if (mode != 2)
            for (j = 0; j < n; ++j) {
                diag[j] = wa2[j];
                if (wa2[j] == 0.)
                    diag[j] = 1.;
            }

        /* on the first iteration, calculate the norm of the scaled x */
        /* and initialize the step bound delta. */

        wa3 = diag.cwise() * x;
        xnorm = wa3.stableNorm();
        delta = parameters.factor * xnorm;
        if (delta == 0.)
            delta = parameters.factor;
    }

    /* form (q transpose)*fvec and store the first n components in */
    /* qtf. */

    wa4 = fvec;
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) != 0.) {
            sum = 0.;
            for (i = j; i < m; ++i)
                sum += fjac(i,j) * wa4[i];
            temp = -sum / fjac(j,j);
            for (i = j; i < m; ++i)
                wa4[i] += fjac(i,j) * temp;
        }
        fjac(j,j) = wa1[j];
        qtf[j] = wa4[j];
    }

    /* compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm != 0.)
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            if (wa2[l] != 0.) {
                sum = 0.;
                for (i = 0; i <= j; ++i)
                    sum += fjac(i,j) * (qtf[i] / fnorm);
                /* Computing MAX */
                gnorm = std::max(gnorm, ei_abs(sum / wa2[l]));
            }
        }

    /* test for convergence of the gradient norm. */

    if (gnorm <= parameters.gtol)
        return CosinusTooSmall;

    /* rescale if necessary. */

    if (mode != 2) /* Computing MAX */
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */
    do {

        /* determine the levenberg-marquardt parameter. */

        ei_lmpar<Scalar>(fjac, ipvt, diag, qtf, delta, par, wa1);

        /* store the direction p and x + p. calculate the norm of p. */

        wa1 = -wa1;
        wa2 = x + wa1;
        wa3 = diag.cwise() * wa1;
        pnorm = wa3.stableNorm();

        /* on the first iteration, adjust the initial step bound. */

        if (iter == 1)
            delta = std::min(delta,pnorm);

        /* evaluate the function at x + p and calculate its norm. */

        if ( functor(wa2, wa4) < 0)
            return UserAsked;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (Scalar(.1) * fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction and */
        /* the scaled directional derivative. */

        wa3.fill(0.);
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            temp = wa1[l];
            for (i = 0; i <= j; ++i)
                wa3[i] += fjac(i,j) * temp;
        }
        temp1 = ei_abs2(wa3.stableNorm() / fnorm);
        temp2 = ei_abs2(ei_sqrt(par) * pnorm / fnorm);
        /* Computing 2nd power */
        prered = temp1 + temp2 / Scalar(.5);
        dirder = -(temp1 + temp2);

        /* compute the ratio of the actual to the predicted */
        /* reduction. */

        ratio = 0.;
        if (prered != 0.)
            ratio = actred / prered;

        /* update the step bound. */

        if (ratio <= Scalar(.25)) {
            if (actred >= 0.)
                temp = Scalar(.5);
            if (actred < 0.)
                temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
            if (Scalar(.1) * fnorm1 >= fnorm || temp < Scalar(.1))
                temp = Scalar(.1);
            /* Computing MIN */
            delta = temp * std::min(delta, pnorm / Scalar(.1));
            par /= temp;
        } else if (!(par != 0. && ratio < Scalar(.75))) {
            delta = pnorm / Scalar(.5);
            par = Scalar(.5) * par;
        }

        /* test for successful iteration. */

        if (ratio >= Scalar(1e-4)) {
            /* successful iteration. update x, fvec, and their norms. */
            x = wa2;
            wa2 = diag.cwise() * x;
            fvec = wa4;
            xnorm = wa2.stableNorm();
            fnorm = fnorm1;
            ++iter;
        }

        /* tests for convergence. */

        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1. && delta <= parameters.xtol * xnorm)
            return RelativeErrorAndReductionTooSmall;
        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1.)
            return RelativeReductionTooSmall;
        if (delta <= parameters.xtol * xnorm)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.)
            return FtolTooSmall;
        if (delta <= epsilon<Scalar>() * xnorm)
            return XtolTooSmall;
        if (gnorm <= epsilon<Scalar>())
            return GtolTooSmall;
        /* end of the inner loop. repeat if iteration unsuccessful. */
    } while (ratio < Scalar(1e-4));
    /* end of the outer loop. */
    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::lmstr1(
        FVectorType  &x,
        const Scalar tol
        )
{
    n = x.size();
    m = functor.values();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return ImproperInputParameters;

    resetParameters();
    parameters.ftol = tol;
    parameters.xtol = tol;
    parameters.maxfev = 100*(n+1);

    return minimizeOptimumStorage(x);
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOptimumStorageInit(
        FVectorType  &x,
        const int mode
        )
{
    n = x.size();
    m = functor.values();

    wa1.resize(n); wa2.resize(n); wa3.resize(n);
    wa4.resize(m);
    fvec.resize(m);
    ipvt.resize(n);
    fjac.resize(m, n);
    if (mode != 2)
        diag.resize(n);
    assert( (mode!=2 || diag.size()==n) || "When using mode==2, the caller must provide a valid 'diag'");
    qtf.resize(n);

    /* Function Body */
    nfev = 0;
    njev = 0;

    /*     check the input parameters for errors. */

    if (n <= 0 || m < n || parameters.ftol < 0. || parameters.xtol < 0. || parameters.gtol < 0. || parameters.maxfev <= 0 || parameters.factor <= 0.)
        return ImproperInputParameters;

    if (mode == 2)
        for (int j = 0; j < n; ++j)
            if (diag[j] <= 0.)
                return ImproperInputParameters;

    /*     evaluate the function at the starting point */
    /*     and calculate its norm. */

    nfev = 1;
    if ( functor(x, fvec) < 0)
        return UserAsked;
    fnorm = fvec.stableNorm();

    /*     initialize levenberg-marquardt parameter and iteration counter. */

    par = 0.;
    iter = 1;

    return Running;
}


template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOptimumStorageOneStep(
        FVectorType  &x,
        const int mode
        )
{
    int i, j, l;
    bool sing;

    /* compute the qr factorization of the jacobian matrix */
    /* calculated one row at a time, while simultaneously */
    /* forming (q transpose)*fvec and storing the first */
    /* n components in qtf. */

    qtf.fill(0.);
    fjac.fill(0.);
    int rownb = 2;
    for (i = 0; i < m; ++i) {
        if (functor.df(x, wa3, rownb) < 0) return UserAsked;
        temp = fvec[i];
        ei_rwupdt<Scalar>(n, fjac.data(), fjac.rows(), wa3.data(), qtf.data(), &temp, wa1.data(), wa2.data());
        ++rownb;
    }
    ++njev;

    /* if the jacobian is rank deficient, call qrfac to */
    /* reorder its columns and update the components of qtf. */

    sing = false;
    for (j = 0; j < n; ++j) {
        if (fjac(j,j) == 0.) {
            sing = true;
        }
        ipvt[j] = j;
        wa2[j] = fjac.col(j).start(j).stableNorm();
    }
    if (sing) {
        ipvt.cwise()+=1;
        wa2 = fjac.colwise().blueNorm();
        ei_qrfac<Scalar>(n, n, fjac.data(), fjac.rows(), true, ipvt.data(), wa1.data());
        ipvt.cwise()-=1; // qrfac() creates ipvt with fortran convention (1->n), convert it to c (0->n-1)
        for (j = 0; j < n; ++j) {
            if (fjac(j,j) != 0.) {
                sum = 0.;
                for (i = j; i < n; ++i)
                    sum += fjac(i,j) * qtf[i];
                temp = -sum / fjac(j,j);
                for (i = j; i < n; ++i)
                    qtf[i] += fjac(i,j) * temp;
            }
            fjac(j,j) = wa1[j];
        }
    }

    /* on the first iteration and if mode is 1, scale according */
    /* to the norms of the columns of the initial jacobian. */

    if (iter == 1) {
        if (mode != 2)
            for (j = 0; j < n; ++j) {
                diag[j] = wa2[j];
                if (wa2[j] == 0.)
                    diag[j] = 1.;
            }

        /* on the first iteration, calculate the norm of the scaled x */
        /* and initialize the step bound delta. */

        wa3 = diag.cwise() * x;
        xnorm = wa3.stableNorm();
        delta = parameters.factor * xnorm;
        if (delta == 0.)
            delta = parameters.factor;
    }

    /* compute the norm of the scaled gradient. */

    gnorm = 0.;
    if (fnorm != 0.)
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            if (wa2[l] != 0.) {
                sum = 0.;
                for (i = 0; i <= j; ++i)
                    sum += fjac(i,j) * (qtf[i] / fnorm);
                /* Computing MAX */
                gnorm = std::max(gnorm, ei_abs(sum / wa2[l]));
            }
        }

    /* test for convergence of the gradient norm. */

    if (gnorm <= parameters.gtol)
        return CosinusTooSmall;

    /* rescale if necessary. */

    if (mode != 2) /* Computing MAX */
        diag = diag.cwise().max(wa2);

    /* beginning of the inner loop. */
    do {

        /* determine the levenberg-marquardt parameter. */

        ei_lmpar<Scalar>(fjac, ipvt, diag, qtf, delta, par, wa1);

        /* store the direction p and x + p. calculate the norm of p. */

        wa1 = -wa1;
        wa2 = x + wa1;
        wa3 = diag.cwise() * wa1;
        pnorm = wa3.stableNorm();

        /* on the first iteration, adjust the initial step bound. */

        if (iter == 1)
            delta = std::min(delta,pnorm);

        /* evaluate the function at x + p and calculate its norm. */

        if ( functor(wa2, wa4) < 0)
            return UserAsked;
        ++nfev;
        fnorm1 = wa4.stableNorm();

        /* compute the scaled actual reduction. */

        actred = -1.;
        if (Scalar(.1) * fnorm1 < fnorm) /* Computing 2nd power */
            actred = 1. - ei_abs2(fnorm1 / fnorm);

        /* compute the scaled predicted reduction and */
        /* the scaled directional derivative. */

        wa3.fill(0.);
        for (j = 0; j < n; ++j) {
            l = ipvt[j];
            temp = wa1[l];
            for (i = 0; i <= j; ++i)
                wa3[i] += fjac(i,j) * temp;
        }
        temp1 = ei_abs2(wa3.stableNorm() / fnorm);
        temp2 = ei_abs2(ei_sqrt(par) * pnorm / fnorm);
        /* Computing 2nd power */
        prered = temp1 + temp2 / Scalar(.5);
        dirder = -(temp1 + temp2);

        /* compute the ratio of the actual to the predicted */
        /* reduction. */

        ratio = 0.;
        if (prered != 0.)
            ratio = actred / prered;

        /* update the step bound. */

        if (ratio <= Scalar(.25)) {
            if (actred >= 0.)
                temp = Scalar(.5);
            if (actred < 0.)
                temp = Scalar(.5) * dirder / (dirder + Scalar(.5) * actred);
            if (Scalar(.1) * fnorm1 >= fnorm || temp < Scalar(.1))
                temp = Scalar(.1);
            /* Computing MIN */
            delta = temp * std::min(delta, pnorm / Scalar(.1));
            par /= temp;
        } else if (!(par != 0. && ratio < Scalar(.75))) {
            delta = pnorm / Scalar(.5);
            par = Scalar(.5) * par;
        }

        /* test for successful iteration. */

        if (ratio >= Scalar(1e-4)) {
            /* successful iteration. update x, fvec, and their norms. */
            x = wa2;
            wa2 = diag.cwise() * x;
            fvec = wa4;
            xnorm = wa2.stableNorm();
            fnorm = fnorm1;
            ++iter;
        }

        /* tests for convergence. */

        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1. && delta <= parameters.xtol * xnorm)
            return RelativeErrorAndReductionTooSmall;
        if (ei_abs(actred) <= parameters.ftol && prered <= parameters.ftol && Scalar(.5) * ratio <= 1.)
            return RelativeReductionTooSmall;
        if (delta <= parameters.xtol * xnorm)
            return RelativeErrorTooSmall;

        /* tests for termination and stringent tolerances. */

        if (nfev >= parameters.maxfev)
            return TooManyFunctionEvaluation;
        if (ei_abs(actred) <= epsilon<Scalar>() && prered <= epsilon<Scalar>() && Scalar(.5) * ratio <= 1.)
            return FtolTooSmall;
        if (delta <= epsilon<Scalar>() * xnorm)
            return XtolTooSmall;
        if (gnorm <= epsilon<Scalar>())
            return GtolTooSmall;
        /* end of the inner loop. repeat if iteration unsuccessful. */
    } while (ratio < Scalar(1e-4));
    /* end of the outer loop. */
    return Running;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::minimizeOptimumStorage(
        FVectorType  &x,
        const int mode
        )
{
    Status status = minimizeOptimumStorageInit(x, mode);
    while (status==Running)
        status = minimizeOptimumStorageOneStep(x, mode);
    return status;
}

template<typename FunctorType, typename Scalar>
typename LevenbergMarquardt<FunctorType,Scalar>::Status
LevenbergMarquardt<FunctorType,Scalar>::lmdif1(
        FunctorType &functor,
        FVectorType  &x,
        int *nfev,
        const Scalar tol
        )
{
    int n = x.size();
    int m = functor.values();

    /* check the input parameters for errors. */
    if (n <= 0 || m < n || tol < 0.)
        return ImproperInputParameters;

    NumericalDiff<FunctorType> numDiff(functor);
    // embedded LevenbergMarquardt
    LevenbergMarquardt<NumericalDiff<FunctorType> > lm(numDiff);
    lm.parameters.ftol = tol;
    lm.parameters.xtol = tol;
    lm.parameters.maxfev = 200*(n+1);

    Status info = Status(lm.minimize(x));
    if (nfev)
        * nfev = lm.nfev;
    return info;
}

//vim: ai ts=4 sts=4 et sw=4
#endif // EIGEN_LEVENBERGMARQUARDT__H

