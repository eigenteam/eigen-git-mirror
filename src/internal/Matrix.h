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

#ifndef EIGEN_MATRIX_H
#define EIGEN_MATRIX_H

#include "Util.h"
#include "EigenBase.h"
#include "MatrixRef.h"
#include "MatrixStorage.h"

namespace Eigen
{

template<typename _Scalar, int _Rows, int _Cols>
class Matrix : public EigenBase<_Scalar, Matrix<_Scalar, _Rows, _Cols> >,
               public MatrixStorage<_Scalar, _Rows, _Cols>
{
  public:
    friend class EigenBase<_Scalar, Matrix>;
    typedef      EigenBase<_Scalar, Matrix>            Base;
    typedef      MatrixStorage<_Scalar, _Rows, _Cols>  Storage;
    typedef      _Scalar                               Scalar;
    typedef      MatrixRef<Matrix>                     Ref;
    typedef      MatrixAlias<Matrix>                   Alias;
    
    static const int RowsAtCompileTime = _Rows, ColsAtCompileTime = _Cols;
    
    Alias alias();
    
    const Scalar* array() const
    { return Storage::m_array; }
    
    Scalar* array()
    { return Storage::m_array; }
    
  private:
    Ref _ref() const { return Ref(*const_cast<Matrix*>(this)); }

    const Scalar& _read(int row, int col = 0) const
    {
      EIGEN_CHECK_RANGES(*this, row, col);
      return array()[row + col * Storage::_rows()];
    }
    
    Scalar& _write(int row, int col = 0)
    {
      EIGEN_CHECK_RANGES(*this, row, col);
      return array()[row + col * Storage::_rows()];
    }
    
  public:
    template<typename OtherDerived> 
    Matrix& operator=(const EigenBase<Scalar, OtherDerived> &other)
    {
      resize(other.rows(), other.cols());
      return Base::operator=(other);
    }
    
    template<typename OtherDerived>
    Matrix& operator+=(const EigenBase<Scalar, OtherDerived> &other)
    {
      return Base::operator+=(other);
    }
    template<typename OtherDerived>
    Matrix& operator-=(const EigenBase<Scalar, OtherDerived> &other)
    {
      return Base::operator-=(other);
    }
  
    explicit Matrix(int rows = 1, int cols = 1) : Storage(rows, cols) {}
    template<typename OtherDerived>
    Matrix(const EigenBase<Scalar, OtherDerived>& other) : Storage(other.rows(), other.cols())
    {
      *this = other;
    }
    ~Matrix() {}
};

template<typename Scalar, typename Derived>
Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
eval(const EigenBase<Scalar, Derived>& expression)
{
  return Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>(expression);
}

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix) \
typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix; \
typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, DynamicSize, X)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int,                  i)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(float,                f)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(double,               d)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<int>,    ci)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<float>,  cf)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<double>, cd)

} // namespace Eigen

#include "MatrixAlias.h"
#include "MatrixOps.h"
#include "ScalarOps.h"
#include "RowAndCol.h"

#endif // EIGEN_MATRIX_H
