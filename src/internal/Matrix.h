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

#ifndef EI_MATRIX_H
#define EI_MATRIX_H

#include "Util.h"
#include "Object.h"
#include "MatrixRef.h"
#include "MatrixStorage.h"

template<typename _Scalar, int _Rows, int _Cols>
class EiMatrix : public EiObject<_Scalar, EiMatrix<_Scalar, _Rows, _Cols> >,
                 public EiMatrixStorage<_Scalar, _Rows, _Cols>
{
  public:
    friend class EiObject<_Scalar, EiMatrix>;
    typedef      EiObject<_Scalar, EiMatrix>            Base;
    typedef      EiMatrixStorage<_Scalar, _Rows, _Cols> Storage;
    typedef      _Scalar                                Scalar;
    typedef      EiMatrixRef<EiMatrix>                  Ref;
    typedef      EiMatrixConstRef<EiMatrix>             ConstRef;
    friend class EiMatrixRef<EiMatrix>;
    friend class EiMatrixConstRef<EiMatrix>;
    
    static const int RowsAtCompileTime = _Rows, ColsAtCompileTime = _Cols;
    
    const Scalar* array() const
    { return Storage::m_array; }
    
    Scalar* array()
    { return Storage::m_array; }
    
  private:
    Ref _ref() { return Ref(*this); }
    ConstRef _constRef() const { return ConstRef(*this); }
    
    const Scalar& _read(int row, int col) const
    {
      EI_CHECK_RANGES(*this, row, col);
      return array()[row + col * Storage::_rows()];
    }
    
    Scalar& _write(int row, int col)
    {
      EI_CHECK_RANGES(*this, row, col);
      return array()[row + col * Storage::_rows()];
    }
    
  public:
    template<typename OtherDerived> 
    EiMatrix& operator=(const EiObject<Scalar, OtherDerived>& other)
    {
      resize(other.rows(), other.cols());
      return Base::operator=(other);
    }
    
    EiMatrix& operator=(const EiMatrix& other)
    {
      resize(other.rows(), other.cols());
      return Base::operator=(other);
    }
    
    EI_INHERIT_ASSIGNMENT_OPERATOR(EiMatrix, +=)
    EI_INHERIT_ASSIGNMENT_OPERATOR(EiMatrix, -=)
    EI_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(EiMatrix, *=)
    EI_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(EiMatrix, /=)
    
    explicit EiMatrix(int rows = 1, int cols = 1) : Storage(rows, cols) {}
    template<typename OtherDerived>
    EiMatrix(const EiObject<Scalar, OtherDerived>& other) : Storage(other.rows(), other.cols())
    {
      *this = other;
    }
    EiMatrix(const EiMatrix& other) : Storage(other.rows(), other.cols())
    {
      *this = other;
    }
    ~EiMatrix() {}
};

#define EI_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix) \
typedef EiMatrix<Type, Size, Size> EiMatrix##SizeSuffix##TypeSuffix; \
typedef EiMatrix<Type, Size, 1>    EiVector##SizeSuffix##TypeSuffix; \
typedef EiMatrix<Type, 1, Size>    EiRowVector##SizeSuffix##TypeSuffix;

#define EI_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EI_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EI_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EI_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EI_MAKE_TYPEDEFS(Type, TypeSuffix, EiDynamic, X)

EI_MAKE_TYPEDEFS_ALL_SIZES(int,                  i)
EI_MAKE_TYPEDEFS_ALL_SIZES(float,                f)
EI_MAKE_TYPEDEFS_ALL_SIZES(double,               d)
EI_MAKE_TYPEDEFS_ALL_SIZES(std::complex<int>,    ci)
EI_MAKE_TYPEDEFS_ALL_SIZES(std::complex<float>,  cf)
EI_MAKE_TYPEDEFS_ALL_SIZES(std::complex<double>, cd)

#undef EI_MAKE_TYPEDEFS_ALL_SIZES
#undef EI_MAKE_TYPEDEFS

#include "Eval.h"
#include "MatrixOps.h"
#include "ScalarOps.h"

#endif // EI_MATRIX_H
