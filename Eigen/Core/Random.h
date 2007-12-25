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

#ifndef EIGEN_RANDOM_H
#define EIGEN_RANDOM_H

template<typename MatrixType> class Random : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar, Random<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, Random<MatrixType> >;
    
    Random(int rows, int cols) : m_rows(rows), m_cols(cols)
    {
      assert(rows > 0 && cols > 0);
    }
   
  private:
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    const Random& _ref() const { return *this; }
    int _rows() const { return m_rows; }
    int _cols() const { return m_cols; }
    
    Scalar _coeff(int, int) const
    {
      return random<Scalar>();
    }
    
  protected:
    int m_rows, m_cols;
};

template<typename Scalar, typename Derived>
Eval<Random<Derived> > MatrixBase<Scalar, Derived>::random(int rows, int cols)
{
  return Random<Derived>(rows, cols).eval();
}

template<typename Scalar, typename Derived>
Eval<Random<Derived> > MatrixBase<Scalar, Derived>::random(int size)
{
  assert(IsVectorAtCompileTime);
  if(RowsAtCompileTime == 1) return Random<Derived>(1, size).eval();
  else return Random<Derived>(size, 1).eval();
}

template<typename Scalar, typename Derived>
Eval<Random<Derived> > MatrixBase<Scalar, Derived>::random()
{
  return Random<Derived>(RowsAtCompileTime, ColsAtCompileTime).eval();
}

#endif // EIGEN_RANDOM_H
