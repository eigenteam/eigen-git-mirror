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

#ifndef EIGEN_NONLINEAR_MATHFUNCTIONS_H
#define EIGEN_NONLINEAR_MATHFUNCTIONS_H

#include <cminpack.h>

template<typename Functor, typename Scalar>
int ei_hybrd1(
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &x,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &fvec,
        Scalar tol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>())
        )
{
    int lwa = (x.size()*(3*x.size()+13))/2;
    Eigen::Matrix< Scalar, Eigen::Dynamic, 1 > wa(lwa);

    fvec.resize(x.size());
    return hybrd1(Functor::f, 0, x.size(), x.data(), fvec.data(), tol, wa.data(), lwa);
}

template<typename Functor, typename Scalar>
int ei_hybrd(
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &x,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &fvec,
        int &nfev,
        Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic > &fjac,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &R,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &qtf,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &diag,
        int mode=1,
        int nb_of_subdiagonals = -1,
        int nb_of_superdiagonals = -1,
        int maxfev = 2000,
        Scalar factor = Scalar(100.),
        Scalar xtol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>()),
        Scalar epsfcn = Scalar(0.),
        int nprint=0
        )
{
    int n = x.size();
    int lr = (n*(n+1))/2;
    Eigen::Matrix< Scalar, Eigen::Dynamic, 1 > wa1(n), wa2(n), wa3(n), wa4(n);


    if (nb_of_subdiagonals<0) nb_of_subdiagonals = n-1;
    if (nb_of_superdiagonals<0) nb_of_superdiagonals = n-1;
    fvec.resize(n);
    qtf.resize(n);
    R.resize(lr);
    int ldfjac = n;
    fjac.resize(ldfjac, n);
    return hybrd(
            Functor::f, 0,
            n, x.data(), fvec.data(),
            xtol, maxfev,
            nb_of_subdiagonals, nb_of_superdiagonals,
            epsfcn, 
            diag.data(), mode, 
            factor,
            nprint, 
            &nfev,
            fjac.data(), ldfjac,
            R.data(), lr,
            qtf.data(),
            wa1.data(), wa2.data(), wa3.data(), wa4.data()
    );
}

template<typename Functor, typename Scalar>
int ei_hybrj(
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &x,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &fvec,
        int &nfev,
        int &njev,
        Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic > &fjac,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &R,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &qtf,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &diag,
        int mode=1,
        int maxfev = 1000,
        Scalar factor = Scalar(100.),
        Scalar xtol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>()),
        int nprint=0
        )
{
    int n = x.size();
    int lr = (n*(n+1))/2;
    Eigen::Matrix< Scalar, Eigen::Dynamic, 1 > wa1(n), wa2(n), wa3(n), wa4(n);

    fvec.resize(n);
    qtf.resize(n);
    R.resize(lr);
    int ldfjac = n;
    fjac.resize(ldfjac, n);
    return hybrj (
            Functor::f, 0,
            n, x.data(), fvec.data(),
            fjac.data(), ldfjac,
            xtol, maxfev,
            diag.data(), mode, 
            factor,
            nprint, 
            &nfev,
            &njev,
            R.data(), lr,
            qtf.data(),
            wa1.data(), wa2.data(), wa3.data(), wa4.data()
    );
}

template<typename Functor, typename Scalar>
int ei_lmder1(
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &x,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &fvec,
        VectorXi &ipvt,
        Scalar tol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>())
        )
{
    int lwa = 5*x.size()+fvec.size();
    int ldfjac = fvec.size();
    Eigen::Matrix< Scalar, Eigen::Dynamic, 1 > wa(lwa);
    Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic > fjac(ldfjac, x.size());

    ipvt.resize(x.size());
    return lmder1 (
            Functor::f, 0,
            fvec.size(), x.size(), x.data(), fvec.data(),
            fjac.data() , ldfjac,
            tol,
            ipvt.data(),
            wa.data(), lwa
    );
}

template<typename Functor, typename Scalar>
int ei_lmder(
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &x,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &fvec,
        int &nfev,
        int &njev,
        Eigen::Matrix< Scalar, Eigen::Dynamic, Eigen::Dynamic > &fjac,
        VectorXi &ipvt,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &wa1,
        Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >  &diag,
        int mode=1,
        double factor = 100.,
        int maxfev = 400,
        Scalar ftol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>()),
        Scalar xtol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>()),
        Scalar gtol = Scalar(0.),
        int nprint=0
        )
{
    Eigen::Matrix< Scalar, Eigen::Dynamic, 1 >
        qtf(x.size()),
        wa2(x.size()), wa3(x.size()),
        wa4(fvec.size());
    int ldfjac = fvec.size();

    ipvt.resize(x.size());
    wa1.resize(x.size());
    fjac.resize(ldfjac, x.size());
    diag.resize(x.size());
    return lmder (
            Functor::f, 0,
            fvec.size(), x.size(), x.data(), fvec.data(),
            fjac.data() , ldfjac,
            ftol, xtol, gtol, 
            maxfev,
            diag.data(), mode,
            factor,
            nprint,
            &nfev, &njev,
            ipvt.data(),
            qtf.data(),
            wa1.data(), wa2.data(), wa3.data(), wa4.data()
    );
}

#endif // EIGEN_NONLINEAR_MATHFUNCTIONS_H

