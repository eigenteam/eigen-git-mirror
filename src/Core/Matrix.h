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

template<typename _Scalar, int _Rows, int _Cols>
class Matrix : public MatrixBase<_Scalar, Matrix<_Scalar, _Rows, _Cols> >,
               public MatrixStorage<_Scalar, _Rows, _Cols>
{
  public:
    friend class MatrixBase<_Scalar, Matrix>;
    typedef      MatrixBase<_Scalar, Matrix>                Base;
    typedef      MatrixStorage<_Scalar, _Rows, _Cols>   Storage;
    typedef      _Scalar                                Scalar;
    typedef      MatrixRef<Matrix>                      Ref;
    friend class MatrixRef<Matrix>;
    
    const Scalar* data() const
    { return Storage::m_data; }
    
    Scalar* data()
    { return Storage::m_data; }
    
  private:
    static const int _RowsAtCompileTime = _Rows, _ColsAtCompileTime = _Cols;
    
    Ref _ref() const { return Ref(*this); }
    
    const Scalar& _coeff(int row, int col) const
    {
      return data()[row + col * Storage::_rows()];
    }
    
    Scalar& _coeffRef(int row, int col)
    {
      return data()[row + col * Storage::_rows()];
    }
    
  public:
    template<typename OtherDerived> 
    Matrix& operator=(const MatrixBase<Scalar, OtherDerived>& other)
    {
      resize(other.rows(), other.cols());
      return Base::operator=(other);
    }
    
    Matrix& operator=(const Matrix& other)
    {
      resize(other.rows(), other.cols());
      return Base::operator=(other);
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Matrix, +=)
    EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Matrix, -=)
    EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Matrix, *=)
    EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Matrix, /=)
    
    explicit Matrix() : Storage()
    {
      assert(_RowsAtCompileTime > 0 && _ColsAtCompileTime > 0);
    }
    explicit Matrix(int dim) : Storage(dim)
    {
      assert(dim > 0);
      assert((_RowsAtCompileTime == 1
              && (_ColsAtCompileTime == Dynamic || _ColsAtCompileTime == dim))
          || (_ColsAtCompileTime == 1
              && (_RowsAtCompileTime == Dynamic || _RowsAtCompileTime == dim)));
    }
    
    // this constructor is very tricky.
    // When Matrix is a fixed-size vector type of size 2,
    // Matrix(x,y) should mean "construct vector with coefficients x,y".
    // Otherwise, Matrix(x,y) should mean "construct matrix with x rows and y cols".
    // Note that in the case of fixed-size, Storage::Storage(int,int) does nothing,
    // so it is harmless to call it and afterwards we just fill the m_data array
    // with the two coefficients. In the case of dynamic size, Storage::Storage(int,int)
    // does what we want to, so it only remains to add some asserts.
    Matrix(int x, int y) : Storage(x, y)
    {
      if((_RowsAtCompileTime == 1 && _ColsAtCompileTime == 2)
      || (_RowsAtCompileTime == 2 && _ColsAtCompileTime == 1))
      {
        (Storage::m_data)[0] = x;
        (Storage::m_data)[1] = y;
      }
      else
      {
        assert(x > 0 && (_RowsAtCompileTime == Dynamic || _RowsAtCompileTime == x)
            && y > 0 && (_ColsAtCompileTime == Dynamic || _ColsAtCompileTime == y));
      }
    }
    Matrix(const float& x, const float& y)
    {
      assert((_RowsAtCompileTime == 1 && _ColsAtCompileTime == 2)
          || (_RowsAtCompileTime == 2 && _ColsAtCompileTime == 1));
      (Storage::m_data)[0] = x;
      (Storage::m_data)[1] = y;
    }
    Matrix(const double& x, const double& y)
    {
      assert((_RowsAtCompileTime == 1 && _ColsAtCompileTime == 2)
          || (_RowsAtCompileTime == 2 && _ColsAtCompileTime == 1));
      (Storage::m_data)[0] = x;
      (Storage::m_data)[1] = y;
    }
    Matrix(const Scalar& x, const Scalar& y, const Scalar& z)
    {
      assert((_RowsAtCompileTime == 1 && _ColsAtCompileTime == 3)
          || (_RowsAtCompileTime == 3 && _ColsAtCompileTime == 1));
      (Storage::m_data)[0] = x;
      (Storage::m_data)[1] = y;
      (Storage::m_data)[2] = z;
    }
    Matrix(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w)
    {
      assert((_RowsAtCompileTime == 1 && _ColsAtCompileTime == 4)
          || (_RowsAtCompileTime == 4 && _ColsAtCompileTime == 1));
      (Storage::m_data)[0] = x;
      (Storage::m_data)[1] = y;
      (Storage::m_data)[2] = z;
      (Storage::m_data)[3] = w;
    }
    Matrix(const Scalar *data, int rows, int cols);
    Matrix(const Scalar *data, int size);
    explicit Matrix(const Scalar *data);
    
    template<typename OtherDerived>
    Matrix(const MatrixBase<Scalar, OtherDerived>& other)
             : Storage(other.rows(), other.cols())
    {
      *this = other;
    }
    Matrix(const Matrix& other) : Storage(other.rows(), other.cols())
    {
      *this = other;
    }
    ~Matrix() {}
};

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix) \
typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix; \
typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix; \
typedef Matrix<Type, 1, Size>    RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int,                  i)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(float,                f)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(double,               d)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<float>,  cf)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<double>, cd)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS

#define EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, SizeSuffix) \
using Eigen::Matrix##SizeSuffix##TypeSuffix; \
using Eigen::Vector##SizeSuffix##TypeSuffix; \
using Eigen::RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(TypeSuffix) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 2) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 3) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 4) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, X)

#define EIGEN_USING_MATRIX_TYPEDEFS \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(i) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(f) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(d) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cf) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cd)

#endif // EIGEN_MATRIX_H
