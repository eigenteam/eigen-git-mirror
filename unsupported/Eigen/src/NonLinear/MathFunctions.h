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
// TODO : fixe Scalar here
int ei_hybrd1(
        Eigen::Matrix< double, Eigen::Dynamic, 1 >  &x,
        Eigen::Matrix< double, Eigen::Dynamic, 1 >  &fvec,
        Scalar tol = Eigen::ei_sqrt(Eigen::machine_epsilon<Scalar>())
        )
{
    int lwa = (x.size()*(3*x.size()+13))/2;
    Eigen::Matrix< double, Eigen::Dynamic, 1 > wa(lwa);
    fvec.resize(x.size());
    return hybrd1(Functor::f, 0, x.size(), x.data(), fvec.data(), tol, wa.data(), lwa);
}

#endif // EIGEN_NONLINEAR_MATHFUNCTIONS_H

