// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SPARSEUTIL_H
#define EIGEN_SPARSEUTIL_H

#ifdef NDEBUG
#define EIGEN_DBG_SPARSE(X)
#else
#define EIGEN_DBG_SPARSE(X) X
#endif

template<typename _Scalar, int _Flags = 0> class SparseMatrix;
template<typename _Scalar, int _Flags = 0> class HashMatrix;
template<typename _Scalar, int _Flags = 0> class LinkedVectorMatrix;

const int AccessPatternNotSupported = 0x0;
const int AccessPatternSupported    = 0x1;


template<typename MatrixType, int AccessPattern> struct ei_support_access_pattern
{
  enum { ret = (int(ei_traits<MatrixType>::SupportedAccessPatterns) & AccessPattern) == AccessPattern
             ? AccessPatternSupported
             : AccessPatternNotSupported
  };
};

template<typename T> class ei_eval<T,Sparse>
{
    typedef typename ei_traits<T>::Scalar _Scalar;
    enum {
          _Flags = ei_traits<T>::Flags
    };

  public:
    typedef SparseMatrix<_Scalar, _Flags> type;
};

#endif // EIGEN_SPARSEUTIL_H
