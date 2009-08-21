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

template<typename Functor, typename Scalar>
int ei_lmstr(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        int &nfev,
        int &njev,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        VectorXi &ipvt,
        Matrix< Scalar, Dynamic, 1 >  &diag,
        int mode=1,
        Scalar factor = 100.,
        int maxfev = 400,
        Scalar ftol = ei_sqrt(epsilon<Scalar>()),
        Scalar xtol = ei_sqrt(epsilon<Scalar>()),
        Scalar gtol = Scalar(0.),
        int nprint=0
        )
{
    Matrix< Scalar, Dynamic, 1 >
        qtf(x.size()),
        wa1(x.size()), wa2(x.size()), wa3(x.size()),
        wa4(fvec.size());
    int ldfjac = fvec.size();

    ipvt.resize(x.size());
    fjac.resize(ldfjac, x.size());
    diag.resize(x.size());
    return lmstr_template<Scalar> (
            Functor::f, 0,
            fvec.size(), x.size(), x.data(), fvec.data(),
            fjac.data() , ldfjac,
            ftol, xtol, gtol, 
            maxfev,
            diag.data(), mode,
            factor,
            nprint,
            nfev, njev,
            ipvt.data(),
            qtf.data(),
            wa1.data(), wa2.data(), wa3.data(), wa4.data()
    );
}

template<typename Functor, typename Scalar>
int ei_lmdif(
        Matrix< Scalar, Dynamic, 1 >  &x,
        Matrix< Scalar, Dynamic, 1 >  &fvec,
        int &nfev,
        Matrix< Scalar, Dynamic, Dynamic > &fjac,
        VectorXi &ipvt,
        Matrix< Scalar, Dynamic, 1 >  &qtf,
        Matrix< Scalar, Dynamic, 1 >  &diag,
        int mode=1,
        Scalar factor = 100.,
        int maxfev = 400,
        Scalar ftol = ei_sqrt(epsilon<Scalar>()),
        Scalar xtol = ei_sqrt(epsilon<Scalar>()),
        Scalar gtol = Scalar(0.),
        Scalar epsfcn = Scalar(0.),
        int nprint=0
        )
{
    Matrix< Scalar, Dynamic, 1 >
        wa1(x.size()), wa2(x.size()), wa3(x.size()),
        wa4(fvec.size());
    int ldfjac = fvec.size();

    ipvt.resize(x.size());
    fjac.resize(ldfjac, x.size());
    diag.resize(x.size());
    qtf.resize(x.size());
    return lmdif_template<Scalar> (
            Functor::f, 0,
            fvec.size(), x.size(), x.data(), fvec.data(),
            ftol, xtol, gtol, 
            maxfev,
            epsfcn,
            diag.data(), mode,
            factor,
            nprint,
            nfev,
            fjac.data() , ldfjac,
            ipvt.data(),
            qtf.data(),
            wa1.data(), wa2.data(), wa3.data(), wa4.data()
    );
}

#endif // EIGEN_NONLINEAR_MATHFUNCTIONS_H

