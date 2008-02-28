// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_MATRIX_H
#define EIGEN_MATRIX_H

template<typename T, int Size> class Array
{
    T m_data[Size];
  public:
    Array() {}
    explicit Array(int) {}
    void resize(int) {}
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

template<typename T> class Array<T, Dynamic>
{
    T *m_data;
  public:
    explicit Array(int size) : m_data(new T[size]) {}
    ~Array() { delete[] m_data; }
    void resize(int size)
    {
      delete[] m_data;
      m_data = new T[size];
    }
    const T *data() const { return m_data; }
    T *data() { return m_data; }
};

/** \class Matrix
  *
  * \brief The matrix class, also used for vectors and row-vectors
  *
  * \param _Scalar the scalar type, i.e. the type of the coefficients
  * \param _Rows the number of rows at compile-time. Use the special value \a Dynamic to specify that the number of rows is dynamic, i.e. is not fixed at compile-time.
  * \param _Cols the number of columns at compile-time. Use the special value \a Dynamic to specify that the number of columns is dynamic, i.e. is not fixed at compile-time.
  * \param _StorageOrder can be either \a RowMajor or \a ColumnMajor.
  * This template parameter has a default value (EIGEN_DEFAULT_MATRIX_STORAGE_ORDER)
  * which, if not predefined, is defined to \a ColumnMajor. You can override this behavior by
  * predefining it before including Eigen headers.
  *
  * This single class template covers all kinds of matrix and vectors that Eigen can handle.
  * All matrix and vector types are just typedefs to specializations of this class template.
  *
  * These typedefs are as follows:
  * \li \c %Matrix\#\#Size\#\#Type for square matrices
  * \li \c Vector\#\#Size\#\#Type for vectors (matrices with one column)
  * \li \c RowVector\#\#Size\#\#Type for row-vectors (matrices with one row)
  *
  * where \c Size can be
  * \li \c 2 for fixed size 2
  * \li \c 3 for fixed size 3
  * \li \c 4 for fixed size 4
  * \li \c X for dynamic size
  *
  * and \c Type can be
  * \li \c i for type \c int
  * \li \c f for type \c float
  * \li \c d for type \c double
  * \li \c cf for type \c std::complex<float>
  * \li \c cd for type \c std::complex<double>
  *
  * Examples:
  * \li \c Matrix2d is a typedef for \c Matrix<double,2,2>
  * \li \c VectorXf is a typedef for \c Matrix<float,Dynamic,1>
  * \li \c RowVector3i is a typedef for \c Matrix<int,1,3>
  *
  * Of course these typedefs do not exhaust all the possibilities offered by the Matrix class
  * template, they only address some of the most common cases. For instance, if you want a
  * fixed-size matrix with 3 rows and 5 columns, there is no typedef for that, so you should use
  * \c Matrix<double,3,5>.
  *
  * Note that most of the API is in the base class MatrixBase.
  */
template<typename _Scalar, int _Rows, int _Cols,
         int _StorageOrder = EIGEN_DEFAULT_MATRIX_STORAGE_ORDER,
         int _MaxRows = _Rows, int _MaxCols = _Cols>
class Matrix : public MatrixBase<_Scalar, Matrix<_Scalar, _Rows, _Cols,
                                                 _StorageOrder, _MaxRows, _MaxCols> >
{
  public:
    friend class MatrixBase<_Scalar, Matrix>;
    friend class Map<Matrix>;
    
    typedef      MatrixBase<_Scalar, Matrix>            Base;
    typedef      _Scalar                                Scalar;
    typedef      MatrixRef<Matrix>                      Ref;
    friend class MatrixRef<Matrix>;
        
  private:
    enum {
      RowsAtCompileTime = _Rows,
      ColsAtCompileTime = _Cols,
      StorageOrder = _StorageOrder,
      MaxRowsAtCompileTime = _MaxRows,
      MaxColsAtCompileTime = _MaxCols,
      MaxSizeAtCompileTime = _MaxRows == Dynamic || _MaxCols == Dynamic
                               ? Dynamic
                               : _MaxRows * _MaxCols
    };

    IntAtRunTimeIfDynamic<RowsAtCompileTime> m_rows;
    IntAtRunTimeIfDynamic<ColsAtCompileTime> m_cols;
    Array<Scalar, MaxSizeAtCompileTime> m_array;

    Ref _ref() const { return Ref(*this); }
    int _rows() const { return m_rows.value(); }
    int _cols() const { return m_cols.value(); }
    
    const Scalar& _coeff(int row, int col) const
    {
      if(StorageOrder == ColumnMajor)
        return m_array.data()[row + col * m_rows.value()];
      else // RowMajor
        return m_array.data()[col + row * m_cols.value()];
    }
    
    Scalar& _coeffRef(int row, int col)
    {
      if(StorageOrder == ColumnMajor)
        return m_array.data()[row + col * m_rows.value()];
      else // RowMajor
        return m_array.data()[col + row * m_cols.value()];
    }
    
  public:
    /** \returns a const pointer to the data array of this matrix */
    const Scalar *data() const
    { return m_array.data(); }
    
    /** \returns a pointer to the data array of this matrix */
    Scalar *data()
    { return m_array.data(); }

    void resize(int rows, int cols)
    {
      assert(rows > 0
          && (MaxRowsAtCompileTime == Dynamic || MaxRowsAtCompileTime >= rows)
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (MaxColsAtCompileTime == Dynamic || MaxColsAtCompileTime >= cols)
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
      m_rows.setValue(rows);
      m_cols.setValue(cols);
      m_array.resize(rows * cols);
    }

    /** Copies the value of the expression \a other into *this.
      *
      * *this is resized (if possible) to match the dimensions of \a other.
      *
      * As a special exception, copying a row-vector into a vector (and conversely)
      * is allowed. The resizing, if any, is then done in the appropriate way so that
      * row-vectors remain row-vectors and vectors remain vectors.
      */
    template<typename OtherDerived> 
    Matrix& operator=(const MatrixBase<Scalar, OtherDerived>& other)
    {
      if(RowsAtCompileTime == 1)
      {
        assert(other.isVector());
        resize(1, other.size());
      }
      else if(ColsAtCompileTime == 1)
      {
        assert(other.isVector());
        resize(other.size(), 1);
      }
      else resize(other.rows(), other.cols());
      return Base::operator=(other);
    }
    
    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    Matrix& operator=(const Matrix& other)
    {
      return operator=<Matrix>(other);
    }
    
    EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Matrix, +=)
    EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Matrix, -=)
    EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Matrix, *=)
    EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Matrix, /=)
    
    static const Map<Matrix> map(const Scalar* array, int rows, int cols);
    static const Map<Matrix> map(const Scalar* array, int size);
    static const Map<Matrix> map(const Scalar* array);
    static Map<Matrix> map(Scalar* array, int rows, int cols);
    static Map<Matrix> map(Scalar* array, int size);
    static Map<Matrix> map(Scalar* array);
    
    /** Default constructor, does nothing. Only for fixed-size matrices.
      * For dynamic-size matrices and vectors, this constructor is forbidden (guarded by
      * an assertion) because it would leave the matrix without an allocated data buffer.
      */
    explicit Matrix()
    {
      assert(RowsAtCompileTime > 0 && ColsAtCompileTime > 0);
    }
    
    /** Constructs a vector or row-vector with given dimension. \only_for_vectors
      *
      * Note that this is only useful for dynamic-size vectors. For fixed-size vectors,
      * it is redundant to pass the dimension here, so it makes more sense to use the default
      * constructor Matrix() instead.
      */
    explicit Matrix(int dim) : m_rows(RowsAtCompileTime == 1 ? 1 : dim),
                               m_cols(ColsAtCompileTime == 1 ? 1 : dim),
                               m_array(dim)
    {
      assert(dim > 0);
      assert((RowsAtCompileTime == 1
              && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == dim))
          || (ColsAtCompileTime == 1
              && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == dim)));
    }
    
    /** This constructor has two very different behaviors, depending on the type of *this.
      *
      * \li When Matrix is a fixed-size vector type of size 2, this constructor constructs
      *     an initialized vector. The parameters \a x, \a y are copied into the first and second
      *     coords of the vector respectively.
      * \li Otherwise, this constructor constructs an uninitialized matrix with \a x rows and
      *     \a y columns. This is useful for dynamic-size matrices. For fixed-size matrices,
      *     it is redundant to pass these parameters, so one should use the default constructor
      *     Matrix() instead.
      */
    Matrix(int x, int y) : m_rows(x), m_cols(y), m_array(x*y)
    {
      if((RowsAtCompileTime == 1 && ColsAtCompileTime == 2)
      || (RowsAtCompileTime == 2 && ColsAtCompileTime == 1))
      {
        m_array.data()[0] = x;
        m_array.data()[1] = y;
      }
      else
      {
        assert(x > 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == x)
            && y > 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == y));
      }
    }
    /** constructs an initialized 2D vector with given coefficients */
    Matrix(const float& x, const float& y)
    {
      assert((RowsAtCompileTime == 1 && ColsAtCompileTime == 2)
          || (RowsAtCompileTime == 2 && ColsAtCompileTime == 1));
      m_array.data()[0] = x;
      m_array.data()[1] = y;
    }
    /** constructs an initialized 2D vector with given coefficients */
    Matrix(const double& x, const double& y)
    {
      assert((RowsAtCompileTime == 1 && ColsAtCompileTime == 2)
          || (RowsAtCompileTime == 2 && ColsAtCompileTime == 1));
      m_array.data()[0] = x;
      m_array.data()[1] = y;
    }
    /** constructs an initialized 3D vector with given coefficients */
    Matrix(const Scalar& x, const Scalar& y, const Scalar& z)
    {
      assert((RowsAtCompileTime == 1 && ColsAtCompileTime == 3)
          || (RowsAtCompileTime == 3 && ColsAtCompileTime == 1));
      m_array.data()[0] = x;
      m_array.data()[1] = y;
      m_array.data()[2] = z;
    }
    /** constructs an initialized 4D vector with given coefficients */
    Matrix(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w)
    {
      assert((RowsAtCompileTime == 1 && ColsAtCompileTime == 4)
          || (RowsAtCompileTime == 4 && ColsAtCompileTime == 1));
      m_array.data()[0] = x;
      m_array.data()[1] = y;
      m_array.data()[2] = z;
      m_array.data()[3] = w;
    }
    Matrix(const Scalar *data, int rows, int cols);
    Matrix(const Scalar *data, int size);
    explicit Matrix(const Scalar *data);
    
    /** Constructor copying the value of the expression \a other */
    template<typename OtherDerived>
    Matrix(const MatrixBase<Scalar, OtherDerived>& other)
             : m_rows(other.rows()),
               m_cols(other.cols()),
               m_array(other.rows() * other.cols())
    {
      *this = other;
    }
    /** Copy constructor */
    Matrix(const Matrix& other)
             : m_rows(other.rows()),
               m_cols(other.cols()),
               m_array(other.rows() * other.cols())
    {
      *this = other;
    }
    /** Destructor */
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
