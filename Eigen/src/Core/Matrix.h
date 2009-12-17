// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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
  * The %Matrix class is the work-horse for all \em dense (\ref dense "note") matrices and vectors within Eigen.
  * Vectors are matrices with one column, and row-vectors are matrices with one row.
  *
  * The %Matrix class encompasses \em both fixed-size and dynamic-size objects (\ref fixedsize "note").
  *
  * The first three template parameters are required:
  * \param _Scalar Numeric type, i.e. float, double, int
  * \param _Rows Number of rows, or \b Dynamic
  * \param _Cols Number of columns, or \b Dynamic
  *
  * The remaining template parameters are optional -- in most cases you don't have to worry about them.
  * \param _Options A combination of either \b RowMajor or \b ColMajor, and of either
  *                 \b AutoAlign or \b DontAlign.
  *                 The former controls storage order, and defaults to column-major. The latter controls alignment, which is required
  *                 for vectorization. It defaults to aligning matrices except for fixed sizes that aren't a multiple of the packet size.
  * \param _MaxRows Maximum number of rows. Defaults to \a _Rows (\ref maxrows "note").
  * \param _MaxCols Maximum number of columns. Defaults to \a _Cols (\ref maxrows "note").
  *
  * Eigen provides a number of typedefs covering the usual cases. Here are some examples:
  *
  * \li \c Matrix2d is a 2x2 square matrix of doubles (\c Matrix<double, 2, 2>)
  * \li \c Vector4f is a vector of 4 floats (\c Matrix<float, 4, 1>)
  * \li \c RowVector3i is a row-vector of 3 ints (\c Matrix<int, 1, 3>)
  *
  * \li \c MatrixXf is a dynamic-size matrix of floats (\c Matrix<float, Dynamic, Dynamic>)
  * \li \c VectorXf is a dynamic-size vector of floats (\c Matrix<float, Dynamic, 1>)
  *
  * \li \c Matrix2Xf is a partially fixed-size (dynamic-size) matrix of floats (\c Matrix<float, 2, Dynamic>)
  * \li \c MatrixX3d is a partially dynamic-size (fixed-size) matrix of double (\c Matrix<double, Dynamic, 3>)
  *
  * See \link matrixtypedefs this page \endlink for a complete list of predefined \em %Matrix and \em Vector typedefs.
  *
  * You can access elements of vectors and matrices using normal subscripting:
  *
  * \code
  * Eigen::VectorXd v(10);
  * v[0] = 0.1;
  * v[1] = 0.2;
  * v(0) = 0.3;
  * v(1) = 0.4;
  *
  * Eigen::MatrixXi m(10, 10);
  * m(0, 1) = 1;
  * m(0, 2) = 2;
  * m(0, 3) = 3;
  * \endcode
  *
  * <i><b>Some notes:</b></i>
  *
  * <dl>
  * <dt><b>\anchor dense Dense versus sparse:</b></dt>
  * <dd>This %Matrix class handles dense, not sparse matrices and vectors. For sparse matrices and vectors, see the Sparse module.
  *
  * Dense matrices and vectors are plain usual arrays of coefficients. All the coefficients are stored, in an ordinary contiguous array.
  * This is unlike Sparse matrices and vectors where the coefficients are stored as a list of nonzero coefficients.</dd>
  *
  * <dt><b>\anchor fixedsize Fixed-size versus dynamic-size:</b></dt>
  * <dd>Fixed-size means that the numbers of rows and columns are known are compile-time. In this case, Eigen allocates the array
  * of coefficients as a fixed-size array, as a class member. This makes sense for very small matrices, typically up to 4x4, sometimes up
  * to 16x16. Larger matrices should be declared as dynamic-size even if one happens to know their size at compile-time.
  *
  * Dynamic-size means that the numbers of rows or columns are not necessarily known at compile-time. In this case they are runtime
  * variables, and the array of coefficients is allocated dynamically on the heap.
  *
  * Note that \em dense matrices, be they Fixed-size or Dynamic-size, <em>do not</em> expand dynamically in the sense of a std::map.
  * If you want this behavior, see the Sparse module.</dd>
  *
  * <dt><b>\anchor maxrows _MaxRows and _MaxCols:</b></dt>
  * <dd>In most cases, one just leaves these parameters to the default values.
  * These parameters mean the maximum size of rows and columns that the matrix may have. They are useful in cases
  * when the exact numbers of rows and columns are not known are compile-time, but it is known at compile-time that they cannot
  * exceed a certain value. This happens when taking dynamic-size blocks inside fixed-size matrices: in this case _MaxRows and _MaxCols
  * are the dimensions of the original matrix, while _Rows and _Cols are Dynamic.</dd>
  * </dl>
  *
  * \see MatrixBase for the majority of the API methods for matrices
  */
template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ei_traits<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
  typedef _Scalar Scalar;
  typedef Dense StorageType;
  enum {
    RowsAtCompileTime = _Rows,
    ColsAtCompileTime = _Cols,
    MaxRowsAtCompileTime = _MaxRows,
    MaxColsAtCompileTime = _MaxCols,
    Flags = ei_compute_matrix_flags<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::ret,
    CoeffReadCost = NumTraits<Scalar>::ReadCost
  };
};

template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
class Matrix
  : public DenseStorageBase<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, MatrixBase, _Options>
{
  public:

    typedef DenseStorageBase<Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>, MatrixBase, _Options> Base;
    _EIGEN_GENERIC_PUBLIC_INTERFACE(Matrix)

    enum { Options = _Options };
    typedef typename Base::PlainMatrixType PlainMatrixType;
    friend class Eigen::Map<Matrix, Unaligned>;
    typedef class Eigen::Map<Matrix, Unaligned> UnalignedMapType;
    friend class Eigen::Map<Matrix, Aligned>;
    typedef class Eigen::Map<Matrix, Aligned> AlignedMapType;

  protected:
    using Base::m_storage;
//     ei_matrix_storage<Scalar, MaxSizeAtCompileTime, RowsAtCompileTime, ColsAtCompileTime, Options> m_storage;

  public:
    enum { NeedsToAlign = (!(Options&DontAlign))
                          && SizeAtCompileTime!=Dynamic && ((sizeof(Scalar)*SizeAtCompileTime)%16)==0 };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)

    using Base::base;
    using Base::coeff;
    using Base::coeffRef;


    /** Copies the value of the expression \a other into \c *this with automatic resizing.
      *
      * *this might be resized to match the dimensions of \a other. If *this was a null matrix (not already initialized),
      * it will be initialized.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Matrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return Base::_set(other);
    }

    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    EIGEN_STRONG_INLINE Matrix& operator=(const Matrix& other)
    {
      return Base::_set(other);
    }

    /** \sa MatrixBase::lazyAssign() */
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE Matrix& lazyAssign(const MatrixBase<OtherDerived>& other)
//     {
//       _resize_to_match(other);
//       return Base::lazyAssign(other.derived());
//     }
//
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE Matrix& operator=(const ReturnByValue<OtherDerived>& func)
//     {
//       resize(func.rows(), func.cols());
//       return Base::operator=(func);
//     }

    using Base::operator =;
    using Base::operator +=;
    using Base::operator -=;
    using Base::operator *=;
    using Base::operator /=;

    /** Default constructor.
      *
      * For fixed-size matrices, does nothing.
      *
      * For dynamic-size matrices, creates an empty matrix of size 0. Does not allocate any array. Such a matrix
      * is called a null matrix. This constructor is the unique way to create null matrices: resizing
      * a matrix to 0 is not supported.
      *
      * \sa resize(int,int)
      */
    EIGEN_STRONG_INLINE explicit Matrix() : Base()
    {
      Base::_check_template_params();
      EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    // FIXME is it still needed
    /** \internal */
    Matrix(ei_constructor_without_unaligned_array_assert)
      : Base(ei_constructor_without_unaligned_array_assert())
    { Base::_check_template_params(); EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED }
#endif

    /** Constructs a vector or row-vector with given dimension. \only_for_vectors
      *
      * Note that this is only useful for dynamic-size vectors. For fixed-size vectors,
      * it is redundant to pass the dimension here, so it makes more sense to use the default
      * constructor Matrix() instead.
      */
    EIGEN_STRONG_INLINE explicit Matrix(int dim)
      : Base(dim, RowsAtCompileTime == 1 ? 1 : dim, ColsAtCompileTime == 1 ? 1 : dim)
    {
      Base::_check_template_params();
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Matrix)
      ei_assert(dim > 0);
      ei_assert(SizeAtCompileTime == Dynamic || SizeAtCompileTime == dim);
      EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename T0, typename T1>
    EIGEN_STRONG_INLINE Matrix(const T0& x, const T1& y)
    {
      Base::_check_template_params();
      Base::template _init2<T0,T1>(x, y);
    }
    #else
    /** constructs an uninitialized matrix with \a rows rows and \a cols columns.
      *
      * This is useful for dynamic-size matrices. For fixed-size matrices,
      * it is redundant to pass these parameters, so one should use the default constructor
      * Matrix() instead. */
    Matrix(int rows, int cols);
    /** constructs an initialized 2D vector with given coefficients */
    Matrix(const Scalar& x, const Scalar& y);
    #endif

    /** constructs an initialized 3D vector with given coefficients */
    EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z)
    {
      Base::_check_template_params();
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 3)
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
      m_storage.data()[2] = z;
    }
    /** constructs an initialized 4D vector with given coefficients */
    EIGEN_STRONG_INLINE Matrix(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& w)
    {
      Base::_check_template_params();
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 4)
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
      m_storage.data()[2] = z;
      m_storage.data()[3] = w;
    }

    explicit Matrix(const Scalar *data);

    /** Constructor copying the value of the expression \a other */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Matrix(const MatrixBase<OtherDerived>& other)
             : Base(other.rows() * other.cols(), other.rows(), other.cols())
    {
      Base::_check_template_params();
      Base::_set_noalias(other);
    }
    /** Copy constructor */
    EIGEN_STRONG_INLINE Matrix(const Matrix& other)
            : Base(other.rows() * other.cols(), other.rows(), other.cols())
    {
      Base::_check_template_params();
      Base::_set_noalias(other);
    }
    /** Copy constructor with in-place evaluation */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Matrix(const ReturnByValue<OtherDerived>& other)
    {
      Base::_check_template_params();
      Base::resize(other.rows(), other.cols());
      other.evalTo(*this);
    }

    /** Destructor */
    inline ~Matrix() {}

    /** \sa MatrixBase::operator=(const AnyMatrixBase<OtherDerived>&) */
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE Matrix& operator=(const AnyMatrixBase<OtherDerived> &other)
//     {
//       resize(other.derived().rows(), other.derived().cols());
//       Base::operator=(other.derived());
//       return *this;
//     }

    /** \sa MatrixBase::operator=(const AnyMatrixBase<OtherDerived>&) */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Matrix(const AnyMatrixBase<OtherDerived> &other)
      : Base(other.derived().rows() * other.derived().cols(), other.derived().rows(), other.derived().cols())
    {
      Base::_check_template_params();
      Base::resize(other.rows(), other.cols());
      *this = other;
    }

    /** Override MatrixBase::swap() since for dynamic-sized matrices of same type it is enough to swap the
      * data pointers.
      */
    template<typename OtherDerived>
    void swap(MatrixBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other)
    { this->_swap(other.derived()); }

    /** \name Map
      * These are convenience functions returning Map objects. The Map() static functions return unaligned Map objects,
      * while the AlignedMap() functions return aligned Map objects and thus should be called only with 16-byte-aligned
      * \a data pointers.
      *
      * \see class Map
      */
    //@{
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

//     using Base::setConstant;
//     Matrix& setConstant(int size, const Scalar& value);
//     Matrix& setConstant(int rows, int cols, const Scalar& value);
//
//     using Base::setZero;
//     Matrix& setZero(int size);
//     Matrix& setZero(int rows, int cols);
//
//     using Base::setOnes;
//     Matrix& setOnes(int size);
//     Matrix& setOnes(int rows, int cols);
//
//     using Base::setRandom;
//     Matrix& setRandom(int size);
//     Matrix& setRandom(int rows, int cols);

    using Base::setIdentity;
    Matrix& setIdentity(int rows, int cols);

/////////// Geometry module ///////////

    template<typename OtherDerived>
    explicit Matrix(const RotationBase<OtherDerived,ColsAtCompileTime>& r);
    template<typename OtherDerived>
    Matrix& operator=(const RotationBase<OtherDerived,ColsAtCompileTime>& r);

    // allow to extend Matrix outside Eigen
    #ifdef EIGEN_MATRIX_PLUGIN
    #include EIGEN_MATRIX_PLUGIN
    #endif

  private:
    /** \internal Resizes *this in preparation for assigning \a other to it.
      * Takes care of doing all the checking that's needed.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      */
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE void _resize_to_match(const MatrixBase<OtherDerived>& other)
//     {
//       #ifdef EIGEN_NO_AUTOMATIC_RESIZING
//       ei_assert((this->size()==0 || (IsVectorAtCompileTime ? (this->size() == other.size())
//                  : (rows() == other.rows() && cols() == other.cols())))
//         && "Size mismatch. Automatic resizing is disabled because EIGEN_NO_AUTOMATIC_RESIZING is defined");
//       #endif
//       resizeLike(other);
//     }

    /** \internal Copies the value of the expression \a other into \c *this with automatic resizing.
      *
      * *this might be resized to match the dimensions of \a other. If *this was a null matrix (not already initialized),
      * it will be initialized.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      *
      * \sa operator=(const MatrixBase<OtherDerived>&), _set_noalias()
      */
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE Matrix& _set(const MatrixBase<OtherDerived>& other)
//     {
//       _set_selector(other.derived(), typename ei_meta_if<static_cast<bool>(int(OtherDerived::Flags) & EvalBeforeAssigningBit), ei_meta_true, ei_meta_false>::ret());
//       return *this;
//     }
//
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE void _set_selector(const OtherDerived& other, const ei_meta_true&) { _set_noalias(other.eval()); }
//
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE void _set_selector(const OtherDerived& other, const ei_meta_false&) { _set_noalias(other); }

    /** \internal Like _set() but additionally makes the assumption that no aliasing effect can happen (which
      * is the case when creating a new matrix) so one can enforce lazy evaluation.
      *
      * \sa operator=(const MatrixBase<OtherDerived>&), _set()
      */
//     template<typename OtherDerived>
//     EIGEN_STRONG_INLINE Matrix& _set_noalias(const MatrixBase<OtherDerived>& other)
//     {
//       _resize_to_match(other);
//       // the 'false' below means to enforce lazy evaluation. We don't use lazyAssign() because
//       // it wouldn't allow to copy a row-vector into a column-vector.
//       return ei_assign_selector<Matrix,OtherDerived,false>::run(*this, other.derived());
//     }

//     static EIGEN_STRONG_INLINE void _check_template_params()
//     {
//       #ifdef EIGEN_DEBUG_MATRIX_CTOR
//         EIGEN_DEBUG_MATRIX_CTOR(Matrix);
//       #endif
//
//       EIGEN_STATIC_ASSERT(((_Rows >= _MaxRows)
//                         && (_Cols >= _MaxCols)
//                         && (_MaxRows >= 0)
//                         && (_MaxCols >= 0)
//                         && (_Rows <= Dynamic)
//                         && (_Cols <= Dynamic)
//                         && (_MaxRows == _Rows || _Rows==Dynamic)
//                         && (_MaxCols == _Cols || _Cols==Dynamic)
//                         && ((_MaxRows==Dynamic?1:_MaxRows)*(_MaxCols==Dynamic?1:_MaxCols)<Dynamic)
//                         && (_Options & (DontAlign|RowMajor)) == _Options),
//         INVALID_MATRIX_TEMPLATE_PARAMETERS)
//     }


//     template<typename T0, typename T1>
//     EIGEN_STRONG_INLINE void _init2(int rows, int cols, typename ei_enable_if<Base::SizeAtCompileTime!=2,T0>::type* = 0)
//     {
//       ei_assert(rows > 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
//              && cols > 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
//       m_storage.resize(rows*cols,rows,cols);
//       EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
//     }
//     template<typename T0, typename T1>
//     EIGEN_STRONG_INLINE void _init2(const Scalar& x, const Scalar& y, typename ei_enable_if<Base::SizeAtCompileTime==2,T0>::type* = 0)
//     {
//       EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Matrix, 2)
//       m_storage.data()[0] = x;
//       m_storage.data()[1] = y;
//     }

//     template<typename MatrixType, typename OtherDerived, bool SwapPointers>
//     friend struct ei_matrix_swap_impl;
};

// template <typename Derived, typename OtherDerived, bool IsVector>
// struct ei_conservative_resize_like_impl
// {
//   static void run(MatrixBase<Derived>& _this, const MatrixBase<OtherDerived>& other)
//   {
//     if (_this.rows() == other.rows() && _this.cols() == other.cols()) return;
//
//     // Note: Here is space for improvement. Basically, for conservativeResize(int,int),
//     // neither RowsAtCompileTime or ColsAtCompileTime must be Dynamic. If only one of the
//     // dimensions is dynamic, one could use either conservativeResize(int rows, NoChange_t) or
//     // conservativeResize(NoChange_t, int cols). For these methods new static asserts like
//     // EIGEN_STATIC_ASSERT_DYNAMIC_ROWS and EIGEN_STATIC_ASSERT_DYNAMIC_COLS would be good.
//     EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(Derived)
//     EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(OtherDerived)
//
//     typename MatrixBase<Derived>::PlainMatrixType tmp(other);
//     const int common_rows = std::min(tmp.rows(), _this.rows());
//     const int common_cols = std::min(tmp.cols(), _this.cols());
//     tmp.block(0,0,common_rows,common_cols) = _this.block(0,0,common_rows,common_cols);
//     _this.derived().swap(tmp);
//   }
// };
//
// template <typename Derived, typename OtherDerived>
// struct ei_conservative_resize_like_impl<Derived,OtherDerived,true>
// {
//   static void run(MatrixBase<Derived>& _this, const MatrixBase<OtherDerived>& other)
//   {
//     if (_this.rows() == other.rows() && _this.cols() == other.cols()) return;
//
//     // segment(...) will check whether Derived/OtherDerived are vectors!
//     typename MatrixBase<Derived>::PlainMatrixType tmp(other);
//     const int common_size = std::min<int>(_this.size(),tmp.size());
//     tmp.segment(0,common_size) = _this.segment(0,common_size);
//     _this.derived().swap(tmp);
//   }
// };

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

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Dynamic> Matrix##Size##X##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Dynamic, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

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
