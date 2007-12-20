// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_EVAL_H
#define EIGEN_EVAL_H

template<typename Expression> class Eval : NoOperatorEquals,
  public Matrix< typename Expression::Scalar,
                 Expression::RowsAtCompileTime,
                 Expression::ColsAtCompileTime >
{
  public:
    typedef typename Expression::Scalar Scalar;
    typedef Matrix<Scalar, Expression::RowsAtCompileTime, Expression::ColsAtCompileTime> MatrixType;
    typedef Expression Base;
    friend class MatrixBase<Scalar, Expression>;
    
    Eval(const Expression& expression) : MatrixType(expression) {}
};

template<typename Scalar, typename Derived>
Eval<Derived> MatrixBase<Scalar, Derived>::eval() const
{
  return Eval<Derived>(*static_cast<const Derived*>(this));
}

#endif // EIGEN_EVAL_H
