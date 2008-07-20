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


/** \class Matrix
  *
  * \brief The matrix class, also used for vectors and row-vectors
  *
  * \param _Scalar the scalar type, i.e. the type of the coefficients
  * \param _Rows the number of rows at compile-time. Use the special value \a Dynamic to
  *              specify that the number of rows is dynamic, i.e. is not fixed at compile-time.
  * \param _Cols the number of columns at compile-time. Use the special value \a Dynamic to
  *              specify that the number of columns is dynamic, i.e. is not fixed at compile-time.
  * \param _MaxRows the maximum number of rows at compile-time. By default this is equal to \a _Rows.
  *              The most common exception is when you don't know the exact number of rows, but know that
  *              it is smaller than some given value. Then you can set \a _MaxRows to that value, and set
  *              _Rows to \a Dynamic.
  * \param _MaxCols the maximum number of cols at compile-time. By default this is equal to \a _Cols.
  *              The most common exception is when you don't know the exact number of cols, but know that
  *              it is smaller than some given value. Then you can set \a _MaxCols to that value, and set
  *              _Cols to \a Dynamic.
  * \param _Flags allows to control certain features such as storage order. See the \ref flags "list of flags".
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
  * See \ref matrixtypedefs for an explicit list of all matrix typedefs.
  *
  * Of course these typedefs do not exhaust all the possibilities offered by the Matrix class
  * template, they only address some of the most common cases. For instance, if you want a
  * fixed-size matrix with 3 rows and 5 columns, there is no typedef for that, so you should use
  * \c Matrix<double,3,5>.
  *
  * Note that most of the API is in the base class MatrixBase.
  */
template<typename _Scalar, int _Rows, int _Cols, int _MaxRows, int _MaxCols, unsigned int _Flags>
struct ei_traits<Matrix<_Scalar, _Rows, _Cols, _MaxRows, _MaxCols, _Flags> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = _Rows,
    ColsAtCompileTime = _Cols,
    MaxRowsAtCompileTime = _MaxRows,
    MaxColsAtCompileTime = _MaxCols,
    Flags = ei_corrected_matrix_flags<
                _Scalar,
                _Rows, _Cols, _MaxRows, _MaxCols,
                _Flags
            >::ret,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = RandomAccessPattern
  };
};

template<typename _Scalar, int _Rows, int _Cols, int _MaxRows, int _MaxCols, unsigned int _Flags>
class Matrix : public MatrixBase<Matrix<_Scalar, _Rows, _Cols, _MaxRows, _MaxCols, _Flags> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(Matrix)
    friend class Eigen::Map<Matrix, Unaligned>;
    friend class Eigen::Map<Matrix, Aligned>;

  protected:
    ei_matrix_storage<Scalar, MaxSizeAtCompileTime, RowsAtCompileTime, ColsAtCompileTime> m_storage;

  public:

    inline int rows() const { return m_storage.rows(); }
    inline int cols() const { return m_storage.cols(); }

    inline int stride(void) const
    {
      if(Flags & RowMajorBit)
        return m_storage.cols();
      else
        return m_storage.rows();
    }

    inline const Scalar& coeff(int row, int col) const
    {
      if(Flags & RowMajorBit)
        return m_storage.data()[col + row * m_storage.cols()];
      else // column-major
        return m_storage.data()[row + col * m_storage.rows()];
    }

    inline const Scalar& coeff(int index) const
    {
      return m_storage.data()[index];
    }

    inline Scalar& coeffRef(int row, int col)
    {
      if(Flags & RowMajorBit)
        return m_storage.data()[col + row * m_storage.cols()];
      else // column-major
        return m_storage.data()[row + col * m_storage.rows()];
    }

    inline Scalar& coeffRef(int index)
    {
      return m_storage.data()[index];
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      return ei_ploadt<Scalar, LoadMode>
               (m_storage.data() + (Flags & RowMajorBit
                                   ? col + row * m_storage.cols()
                                   : row + col * m_storage.rows()));
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      return ei_ploadt<Scalar, LoadMode>(m_storage.data() + index);
    }

    template<int StoreMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode>
              (m_storage.data() + (Flags & RowMajorBit
                                   ? col + row * m_storage.cols()
                                   : row + col * m_storage.rows()), x);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode>(m_storage.data() + index, x);
    }

  public:
    /** \returns a const pointer to the data array of this matrix */
    inline const Scalar *data() const
    { return m_storage.data(); }

    /** \returns a pointer to the data array of this matrix */
    inline Scalar *data()
    { return m_storage.data(); }

    inline void resize(int rows, int cols)
    {
      ei_assert(rows > 0
          && (MaxRowsAtCompileTime == Dynamic || MaxRowsAtCompileTime >= rows)
          && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
          && cols > 0
          && (MaxColsAtCompileTime == Dynamic || MaxColsAtCompileTime >= cols)
          && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
      m_storage.resize(rows * cols, rows, cols);
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
    inline Matrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      if(RowsAtCompileTime == 1)
      {
        ei_assert(other.isVector());
        resize(1, other.size());
      }
      else if(ColsAtCompileTime == 1)
      {
        ei_assert(other.isVector());
        resize(other.size(), 1);
      }
      else resize(other.rows(), other.cols());
      return Base::operator=(other.derived());
    }

    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    inline Matrix& operator=(const Matrix& other)
    {
      return operator=<Matrix>(other);
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Matrix, +=)
    EIGEN_INHERIT_ASSIGNMENT_OPERATOR(Matrix, -=)
    EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Matrix, *=)
    EIGEN_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Matrix, /=)

    /** Default constructor.
      *
      * For fixed-size matrices, does nothing.
      *
      * For dynamic-size matrices, initializes with initial size 1x1, which is inefficient, hence
      * when performance matters one should avoid using this constructor on dynamic-size matrices.
      */
    inline explicit Matrix() : m_storage(1, 1, 1)
    {
      ei_assert(RowsAtCompileTime > 0 && ColsAtCompileTime > 0);
    }

    /** Constructs a vector or row-vector with given dimension. \only_for_vectors
      *
      * Note that this is only useful for dynamic-size vectors. For fixed-size vectors,
      * it is redundant to pass the dimension here, so it makes more sense to use the default
      * constructor Matrix() instead.
      */
    inline explicit Matrix(int dim)
      : m_storage(dim, RowsAtCompileTime == 1 ? 1 : dim, ColsAtCompileTime == 1 ? 1 : dim)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Matrix)
      ei_assert(dim > 0);
      ei_assert(SizeAtCompileTime == Dynamic || SizeAtCompileTime == dim);
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
    inline Matrix(int x, int y) : m_storage(x*y, x, y)
    {
      if((RowsAtCompileTime == 1 && ColsAtCompileTime == 2)
      || (RowsAtCompileTime == 2 && ColsAtCompileTime == 1))
      {
        m_storage.data()[0] = x;
        m_storage.data()[1] = y;
      }
      else
      {
        ei_assert(x > 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == x)
               && y > 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == y));
      }
    }
    /** constructs an initialized 2D vector with given coefficients */
    inline Matrix(const float& x, const float& y)
    {
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 2);
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
    }
    /** constructs an initialized 2D vector with given coefficients */
    inline Matrix(const double& x, const double& y)
    {
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 2);
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
    }
    /** constructs an initialized 3D vector with given coefficients */
    inline Matrix(const Scalar& x, const Scalar& y, const Scalar& z)
    {
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 3);
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
      m_storage.data()[2] = z;
    }
    /** constructs an initialized 4D vector with given coefficients */
    inline Matrix(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w)
    {
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 4);
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
      m_storage.data()[2] = z;
      m_storage.data()[3] = w;
    }

    explicit Matrix(const Scalar *data);

    /** Constructor copying the value of the expression \a other */
    template<typename OtherDerived>
    inline Matrix(const MatrixBase<OtherDerived>& other)
             : m_storage(other.rows() * other.cols(), other.rows(), other.cols())
    {
      Base::lazyAssign(other.derived());
    }
    /** Copy constructor */
    inline Matrix(const Matrix& other)
            : Base(), m_storage(other.rows() * other.cols(), other.rows(), other.cols())
    {
      Base::lazyAssign(other);
    }
    /** Destructor */
    inline ~Matrix() {}

    /** Override MatrixBase::eval() since matrices don't need to be evaluated, it is enough to just read them.
      * This prevents a useless copy when doing e.g. "m1 = m2.eval()"
      */
    const Matrix& eval() const
    {
      return *this;
    }

    /** Override MatrixBase::swap() since for dynamic-sized matrices of same type it is enough to swap the
      * data pointers.
      */
    void swap(Matrix& other)
    {
      if (Base::SizeAtCompileTime==Dynamic)
        m_storage.swap(other.m_storage);
      else
        this->Base::swap(other);
    }
};

/** \defgroup matrixtypedefs Global matrix typedefs
  * Eigen defines several typedef shortcuts for most common matrix types.
  * Here is the explicit list.
  * \sa class Matrix
  */

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
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

#define EIGEN_MAKE_TYPEDEFS_LARGE(Type, TypeSuffix) \
typedef Matrix<Type, Dynamic, Dynamic, EIGEN_DEFAULT_MATRIX_FLAGS | LargeBit> MatrixXL##TypeSuffix; \
typedef Matrix<Type, Dynamic, 1, EIGEN_DEFAULT_MATRIX_FLAGS | LargeBit>       VectorXL##TypeSuffix; \
typedef Matrix<Type, 1, Dynamic, EIGEN_DEFAULT_MATRIX_FLAGS | LargeBit>       RowVectorXL##TypeSuffix;

EIGEN_MAKE_TYPEDEFS_LARGE(int,                  i)
EIGEN_MAKE_TYPEDEFS_LARGE(float,                f)
EIGEN_MAKE_TYPEDEFS_LARGE(double,               d)
EIGEN_MAKE_TYPEDEFS_LARGE(std::complex<float>,  cf)
EIGEN_MAKE_TYPEDEFS_LARGE(std::complex<double>, cd)

#undef EIGEN_MAKE_TYPEDEFS_LARGE

#define EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, SizeSuffix) \
using Eigen::Matrix##SizeSuffix##TypeSuffix; \
using Eigen::Vector##SizeSuffix##TypeSuffix; \
using Eigen::RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(TypeSuffix) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 2) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 3) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, 4) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, X) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE_AND_SIZE(TypeSuffix, XL)

#define EIGEN_USING_MATRIX_TYPEDEFS \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(i) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(f) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(d) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cf) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cd)

#endif // EIGEN_MATRIX_H
