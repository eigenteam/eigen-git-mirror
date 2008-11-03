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
  * Eigen's matrix class is the work-horse for all \em dense matrices and vectors within Eigen.  Dense
  * matrices may either be allocated on the stack, using the template parameters above, or \em dynamically
  * by specifying \em Dynamic as the size.
  *
  *
  * \param _Scalar Numeric type, i.e. float, double, int
  * \param _Rows Number of rows, or \b Dynamic
  * \param _Cols Number of columns, or \b Dynamic
  * \param _StorageOrder Either RowMajor or ColMajor. The default is ColMajor.
  * \param _MaxRows Maximum number of rows. Defaults to \a _Rows.  See note below.
  * \param _MaxCols Maximum number of columns. Defaults to \a _Cols.  See note below.
  *
  * \note <b>Dynamic size:</b>
  * \em Dynamic in this context only means specified at run-time instead of at compile time.  Dynamic
  * matrices <em>do not</em> expand dynamically.
  *
  * \note <b>Max Rows / Columns:</b>
  * The most common reason to use these values is when you don't know the exact number of columns or rows,
  * but know that they will remain below the given value.  Then you can set the \a _MaxRows or \a _MaxCols
  * to that value, and set \a _Rows or \a _Cols to \a Dynamic.
  *
  * \warning For very large matrices, \em Dynamic allocation should be used, otherwise the stack will be
  * overflowed.
  *
  * Eigen provides a number of typedefs to make working with matrices and vector simpler:
  *
  * For example:
  *
  * \li <b>\c MatrixXf is a dynamically sized matrix of floats (\c Matrix<float, Dynamic, Dynamic>)</b>
  * \li <b>\c VectorXf is a dynamically sized vector of floats (\c Matrix<float, Dynamic, 1>)</b>
  *
  * \li \c Matrix2d is a 2-row by 2-column square matrix of doubles (\c Matrix<double, 2, 2>)
  * \li \c RowVector3i is a row-vector with three elements containing integers (\c Matrix<int, 1, 3>)
  *
  * \see matrixtypedefs for a complete list of predefined \em Matrix and \em Vector types.
  *
  * You can access elements of vectors and matrices using normal subscripting:
  *
  * \code
  *
  * Eigen::VectorXf v(10);
  * v[0] = 0.1;
  * v[1] = 0.2;
  *
  * Eigen::MatrixXi m(10, 10);
  * m(0, 1) = 1;
  * m(0, 2) = 2;
  * m(0, 3) = 3;
  *
  * \endcode
  *
  * \see MatrixBase for the majority of the API methods for matrices
  */
template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols>
struct ei_traits<Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = _Rows,
    ColsAtCompileTime = _Cols,
    MaxRowsAtCompileTime = _MaxRows,
    MaxColsAtCompileTime = _MaxCols,
    Flags = ei_compute_matrix_flags<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols>::ret,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = RandomAccessPattern
  };
};

template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols>
class Matrix
  : public MatrixBase<Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols> >
    #ifdef EIGEN_VECTORIZE
    , public ei_with_aligned_operator_new<_Scalar,ei_size_at_compile_time<_Rows,_Cols>::ret>
    #endif
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(Matrix)
    friend class Eigen::Map<Matrix, Unaligned>;
    typedef class Eigen::Map<Matrix, Unaligned> UnalignedMapType;
    friend class Eigen::Map<Matrix, Aligned>;
    typedef class Eigen::Map<Matrix, Aligned> AlignedMapType;

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

    /** \returns a const pointer to the data array of this matrix */
    inline const Scalar *data() const
    { return m_storage.data(); }

    /** \returns a pointer to the data array of this matrix */
    inline Scalar *data()
    { return m_storage.data(); }

    /** Resizes \c *this to a \a rows x \a cols matrix.
      *
      * Makes sense for dynamic-size matrices only.
      *
      * If the current number of coefficients of \c *this exactly matches the
      * product \a rows * \a cols, then no memory allocation is performed and
      * the current values are left unchanged. In all other cases, including
      * shrinking, the data is reallocated and all previous values are lost.
      *
      * \sa resize(int) for vectors.
      */
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

    /** Resizes \c *this to a vector of length \a size
      *
      * \sa resize(int,int) for the details.
      */
    inline void resize(int size)
    {
      ei_assert(size>0 && "a vector cannot be resized to 0 length");
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Matrix)
      if(RowsAtCompileTime == 1)
        m_storage.resize(size, 1, size);
      else
        m_storage.resize(size, size, 1);
    }

    /** Copies the value of the expression \a other into \c *this.
      *
      * \warning Note that the sizes of \c *this and \a other must match.
      * If you want automatic resizing, then you must use the function set().
      *
      * As a special exception, copying a row-vector into a vector (and conversely)
      * is allowed.
      *
      * \sa set()
      */
    template<typename OtherDerived>
    inline Matrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      ei_assert(m_storage.data()!=0 && "you cannot use operator= with a non initialized matrix (instead use set()");
      return Base::operator=(other.derived());
    }

    /** Copies the value of the expression \a other into \c *this with automatic resizing.
      *
      * This function is the same than the assignment operator = excepted that \c *this might
      * be resized to match the dimensions of \a other.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      *
      * \sa operator=()
      */
    template<typename OtherDerived>
    inline Matrix& set(const MatrixBase<OtherDerived>& other)
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
      * For dynamic-size matrices, creates an empty matrix of size null.
      * \warning while creating such an \em null matrix is allowed, it \b cannot
      * \b be \b used before having being resized or initialized with the function set().
      * In particular, initializing a null matrix with operator = is not supported.
      * Finally, this constructor is the unique way to create null matrices: resizing
      * a matrix to 0 is not supported.
      * Here are some examples:
      * \code
      * MatrixXf r = MatrixXf::Random(3,4); // create a random matrix of floats
      * MatrixXf m1, m2;  // creates two null matrices of float
      *
      * m1 = r;           // illegal (raise an assertion)
      * r = m1;           // illegal (raise an assertion)
      * m1 = m2;          // illegal (raise an assertion)
      * m1.set(r);        // OK
      * m2.resize(3,4);
      * m2 = r;           // OK
      * \endcode
      *
      * \sa resize(int,int), set()
      */
    inline explicit Matrix() : m_storage()
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
        m_storage.data()[0] = Scalar(x);
        m_storage.data()[1] = Scalar(y);
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
      ei_assign_selector<Matrix,OtherDerived,false>::run(*this, other.derived());
      //Base::operator=(other.derived());
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
    inline const Matrix& eval() const
    {
      return *this;
    }

    /** Override MatrixBase::swap() since for dynamic-sized matrices of same type it is enough to swap the
      * data pointers.
      */
    inline void swap(Matrix& other)
    {
      if (Base::SizeAtCompileTime==Dynamic)
        m_storage.swap(other.m_storage);
      else
        this->Base::swap(other);
    }

    /**
      * These are convenience functions returning Map objects. The Map() static functions return unaligned Map objects,
      * while the AlignedMap() functions return aligned Map objects and thus should be called only with 16-byte-aligned
      * \a data pointers.
      * 
      * \see class Map
      */
    //@}
    inline static const UnalignedMapType Map(const Scalar* data)
    { return UnalignedMapType(data); }
    inline static UnalignedMapType Map(Scalar* data)
    { return UnalignedMapType(data); }
    inline static const UnalignedMapType Map(const Scalar* data, int size)
    { return UnalignedMapType(data, size); }
    inline static UnalignedMapType Map(Scalar* data, int size)
    { return UnalignedMapType(data, size); }
    inline static const UnalignedMapType Map(const Scalar* data, int rows, int cols)
    { return UnalignedMapType(data, rows, cols); }
    inline static UnalignedMapType Map(Scalar* data, int rows, int cols)
    { return UnalignedMapType(data, rows, cols); }

    inline static const AlignedMapType MapAligned(const Scalar* data)
    { return AlignedMapType(data); }
    inline static AlignedMapType MapAligned(Scalar* data)
    { return AlignedMapType(data); }
    inline static const AlignedMapType MapAligned(const Scalar* data, int size)
    { return AlignedMapType(data, size); }
    inline static AlignedMapType MapAligned(Scalar* data, int size)
    { return AlignedMapType(data, size); }
    inline static const AlignedMapType MapAligned(const Scalar* data, int rows, int cols)
    { return AlignedMapType(data, rows, cols); }
    inline static AlignedMapType MapAligned(Scalar* data, int rows, int cols)
    { return AlignedMapType(data, rows, cols); }
    //@}

/////////// Geometry module ///////////

    template<typename OtherDerived>
    explicit Matrix(const RotationBase<OtherDerived,ColsAtCompileTime>& r);
    template<typename OtherDerived>
    Matrix& operator=(const RotationBase<OtherDerived,ColsAtCompileTime>& r);

    // allow to extend Matrix outside Eigen
    #ifdef EIGEN_MATRIX_PLUGIN
    #include EIGEN_MATRIX_PLUGIN
    #endif
};

/** \defgroup matrixtypedefs Global matrix typedefs
  *
  * \ingroup Core_Module
  *
  * Eigen defines several typedef shortcuts for most common matrix and vector types.
  *
  * The general patterns are the following:
  *
  * \c MatrixSizeType where \c Size can be \c 2,\c 3,\c 4 for fixed size square matrices or \c X for dynamic size,
  * and where \c Type can be \c i for integer, \c f for float, \c d for double, \c cf for complex float, \c cd
  * for complex double.
  *
  * For example, \c Matrix3d is a fixed-size 3x3 matrix type of doubles, and \c MatrixXf is a dynamic-size matrix of floats.
  *
  * There are also \c VectorSizeType and \c RowVectorSizeType which are self-explanatory. For example, \c Vector4cf is
  * a fixed-size vector of 4 complex floats.
  *
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

#define EIGEN_USING_MATRIX_TYPEDEFS \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(i) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(f) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(d) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cf) \
EIGEN_USING_MATRIX_TYPEDEFS_FOR_TYPE(cd)

#endif // EIGEN_MATRIX_H
