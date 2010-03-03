// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_DENSESTORAGEBASE_H
#define EIGEN_DENSESTORAGEBASE_H

#ifdef EIGEN_INITIALIZE_MATRICES_BY_ZERO
# define EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED for(int i=0;i<base().size();++i) coeffRef(i)=Scalar(0);
#else
# define EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
#endif

template <typename Derived, typename OtherDerived = Derived, bool IsVector = static_cast<bool>(Derived::IsVectorAtCompileTime)> struct ei_conservative_resize_like_impl;
template<typename MatrixTypeA, typename MatrixTypeB, bool SwapPointers> struct ei_matrix_swap_impl;

/**
* \brief Dense storage base class for matrices and arrays.
**/
template<typename Derived, template<typename> class _Base, int _Options>
class DenseStorageBase : public _Base<Derived>
{
  public:
    enum { Options = _Options };
    typedef _Base<Derived> Base;
    typedef typename Base::PlainObject PlainObject;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::PacketScalar PacketScalar;
    using Base::RowsAtCompileTime;
    using Base::ColsAtCompileTime;
    using Base::SizeAtCompileTime;
    using Base::MaxRowsAtCompileTime;
    using Base::MaxColsAtCompileTime;
    using Base::MaxSizeAtCompileTime;
    using Base::IsVectorAtCompileTime;
    using Base::Flags;

    friend  class Eigen::Map<Derived, Unaligned>;
    typedef class Eigen::Map<Derived, Unaligned>  UnalignedMapType;
    friend  class Eigen::Map<Derived, Aligned>;
    typedef class Eigen::Map<Derived, Aligned>    AlignedMapType;

  protected:
    ei_matrix_storage<Scalar, Base::MaxSizeAtCompileTime, Base::RowsAtCompileTime, Base::ColsAtCompileTime, Options> m_storage;

  public:
    enum { NeedsToAlign = (!(Options&DontAlign))
                          && SizeAtCompileTime!=Dynamic && ((sizeof(Scalar)*SizeAtCompileTime)%16)==0 };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)

    Base& base() { return *static_cast<Base*>(this); }
    const Base& base() const { return *static_cast<const Base*>(this); }

    EIGEN_STRONG_INLINE int rows() const { return m_storage.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return m_storage.cols(); }

    EIGEN_STRONG_INLINE const Scalar& coeff(int row, int col) const
    {
      if(Flags & RowMajorBit)
        return m_storage.data()[col + row * m_storage.cols()];
      else // column-major
        return m_storage.data()[row + col * m_storage.rows()];
    }

    EIGEN_STRONG_INLINE const Scalar& coeff(int index) const
    {
      return m_storage.data()[index];
    }

    EIGEN_STRONG_INLINE Scalar& coeffRef(int row, int col)
    {
      if(Flags & RowMajorBit)
        return m_storage.data()[col + row * m_storage.cols()];
      else // column-major
        return m_storage.data()[row + col * m_storage.rows()];
    }

    EIGEN_STRONG_INLINE Scalar& coeffRef(int index)
    {
      return m_storage.data()[index];
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int row, int col) const
    {
      return ei_ploadt<Scalar, LoadMode>
               (m_storage.data() + (Flags & RowMajorBit
                                   ? col + row * m_storage.cols()
                                   : row + col * m_storage.rows()));
    }

    template<int LoadMode>
    EIGEN_STRONG_INLINE PacketScalar packet(int index) const
    {
      return ei_ploadt<Scalar, LoadMode>(m_storage.data() + index);
    }

    template<int StoreMode>
    EIGEN_STRONG_INLINE void writePacket(int row, int col, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode>
              (m_storage.data() + (Flags & RowMajorBit
                                   ? col + row * m_storage.cols()
                                   : row + col * m_storage.rows()), x);
    }

    template<int StoreMode>
    EIGEN_STRONG_INLINE void writePacket(int index, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode>(m_storage.data() + index, x);
    }

    /** \returns a const pointer to the data array of this matrix */
    EIGEN_STRONG_INLINE const Scalar *data() const
    { return m_storage.data(); }

    /** \returns a pointer to the data array of this matrix */
    EIGEN_STRONG_INLINE Scalar *data()
    { return m_storage.data(); }

    /** Resizes \c *this to a \a rows x \a cols matrix.
      *
      * This method is intended for dynamic-size matrices, although it is legal to call it on any
      * matrix as long as fixed dimensions are left unchanged. If you only want to change the number
      * of rows and/or of columns, you can use resize(NoChange_t, int), resize(int, NoChange_t).
      *
      * If the current number of coefficients of \c *this exactly matches the
      * product \a rows * \a cols, then no memory allocation is performed and
      * the current values are left unchanged. In all other cases, including
      * shrinking, the data is reallocated and all previous values are lost.
      *
      * Example: \include Matrix_resize_int_int.cpp
      * Output: \verbinclude Matrix_resize_int_int.out
      *
      * \sa resize(int) for vectors, resize(NoChange_t, int), resize(int, NoChange_t)
      */
    inline void resize(int rows, int cols)
    {
      ei_assert((MaxRowsAtCompileTime == Dynamic || MaxRowsAtCompileTime >= rows)
             && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
             && (MaxColsAtCompileTime == Dynamic || MaxColsAtCompileTime >= cols)
             && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
      #ifdef EIGEN_INITIALIZE_MATRICES_BY_ZERO
        int size = rows*cols;
        bool size_changed = size != this->size();
        m_storage.resize(size, rows, cols);
        if(size_changed) EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
      #else
        m_storage.resize(rows*cols, rows, cols);
      #endif
    }

    /** Resizes \c *this to a vector of length \a size
      *
      * \only_for_vectors. This method does not work for
      * partially dynamic matrices when the static dimension is anything other
      * than 1. For example it will not work with Matrix<double, 2, Dynamic>.
      *
      * Example: \include Matrix_resize_int.cpp
      * Output: \verbinclude Matrix_resize_int.out
      *
      * \sa resize(int,int), resize(NoChange_t, int), resize(int, NoChange_t)
      */
    inline void resize(int size)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(DenseStorageBase)
      ei_assert(SizeAtCompileTime == Dynamic || SizeAtCompileTime == size);
      #ifdef EIGEN_INITIALIZE_MATRICES_BY_ZERO
        bool size_changed = size != this->size();
      #endif
      if(RowsAtCompileTime == 1)
        m_storage.resize(size, 1, size);
      else
        m_storage.resize(size, size, 1);
      #ifdef EIGEN_INITIALIZE_MATRICES_BY_ZERO
        if(size_changed) EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
      #endif
    }

    /** Resizes the matrix, changing only the number of columns. For the parameter of type NoChange_t, just pass the special value \c NoChange
      * as in the example below.
      *
      * Example: \include Matrix_resize_NoChange_int.cpp
      * Output: \verbinclude Matrix_resize_NoChange_int.out
      *
      * \sa resize(int,int)
      */
    inline void resize(NoChange_t, int cols)
    {
      resize(rows(), cols);
    }

    /** Resizes the matrix, changing only the number of rows. For the parameter of type NoChange_t, just pass the special value \c NoChange
      * as in the example below.
      *
      * Example: \include Matrix_resize_int_NoChange.cpp
      * Output: \verbinclude Matrix_resize_int_NoChange.out
      *
      * \sa resize(int,int)
      */
    inline void resize(int rows, NoChange_t)
    {
      resize(rows, cols());
    }

    /** Resizes \c *this to have the same dimensions as \a other.
      * Takes care of doing all the checking that's needed.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE void resizeLike(const DenseBase<OtherDerived>& other)
    {
      if(RowsAtCompileTime == 1)
      {
        ei_assert(other.rows() == 1 || other.cols() == 1);
        resize(1, other.size());
      }
      else if(ColsAtCompileTime == 1)
      {
        ei_assert(other.rows() == 1 || other.cols() == 1);
        resize(other.size(), 1);
      }
      else resize(other.rows(), other.cols());
    }

    /** Resizes \c *this to a \a rows x \a cols matrix while leaving old values of \c *this untouched.
      *
      * This method is intended for dynamic-size matrices. If you only want to change the number
      * of rows and/or of columns, you can use conservativeResize(NoChange_t, int),
      * conservativeResize(int, NoChange_t).
      *
      * The top-left part of the resized matrix will be the same as the overlapping top-left corner
      * of \c *this. In case values need to be appended to the matrix they will be uninitialized.
      */
    EIGEN_STRONG_INLINE void conservativeResize(int rows, int cols)
    {
      ei_conservative_resize_like_impl<Derived>::run(*this, rows, cols);
    }

    EIGEN_STRONG_INLINE void conservativeResize(int rows, NoChange_t)
    {
      // Note: see the comment in conservativeResize(int,int)
      conservativeResize(rows, cols());
    }

    EIGEN_STRONG_INLINE void conservativeResize(NoChange_t, int cols)
    {
      // Note: see the comment in conservativeResize(int,int)
      conservativeResize(rows(), cols);
    }

    /** Resizes \c *this to a vector of length \a size while retaining old values of *this.
      *
      * \only_for_vectors. This method does not work for
      * partially dynamic matrices when the static dimension is anything other
      * than 1. For example it will not work with Matrix<double, 2, Dynamic>.
      *
      * When values are appended, they will be uninitialized.
      */
    EIGEN_STRONG_INLINE void conservativeResize(int size)
    {
      ei_conservative_resize_like_impl<Derived>::run(*this, size);
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE void conservativeResizeLike(const DenseBase<OtherDerived>& other)
    {
      ei_conservative_resize_like_impl<Derived,OtherDerived>::run(*this, other);
    }

    /** This is a special case of the templated operator=. Its purpose is to
      * prevent a default operator= from hiding the templated operator=.
      */
    EIGEN_STRONG_INLINE Derived& operator=(const DenseStorageBase& other)
    {
      return _set(other);
    }

    /** \sa MatrixBase::lazyAssign() */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Derived& lazyAssign(const DenseBase<OtherDerived>& other)
    {
      _resize_to_match(other);
      return Base::lazyAssign(other.derived());
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Derived& operator=(const ReturnByValue<OtherDerived>& func)
    {
      resize(func.rows(), func.cols());
      return Base::operator=(func);
    }

    EIGEN_STRONG_INLINE explicit DenseStorageBase() : m_storage()
    {
//       _check_template_params();
//       EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    // FIXME is it still needed ?
    /** \internal */
    DenseStorageBase(ei_constructor_without_unaligned_array_assert)
      : m_storage(ei_constructor_without_unaligned_array_assert())
    {
//       _check_template_params(); EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
    }
#endif

    EIGEN_STRONG_INLINE DenseStorageBase(int size, int rows, int cols)
      : m_storage(size, rows, cols)
    {
//       _check_template_params();
//       EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
    }

    /** \copydoc MatrixBase::operator=(const EigenBase<OtherDerived>&)
      */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Derived& operator=(const EigenBase<OtherDerived> &other)
    {
      resize(other.derived().rows(), other.derived().cols());
      Base::operator=(other.derived());
      return this->derived();
    }

    /** \sa MatrixBase::operator=(const EigenBase<OtherDerived>&) */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE DenseStorageBase(const EigenBase<OtherDerived> &other)
      : m_storage(other.derived().rows() * other.derived().cols(), other.derived().rows(), other.derived().cols())
    {
      _check_template_params();
      Base::operator=(other.derived());
    }

    /** \name Map
      * These are convenience functions returning Map objects. The Map() static functions return unaligned Map objects,
      * while the AlignedMap() functions return aligned Map objects and thus should be called only with 16-byte-aligned
      * \a data pointers.
      *
      * These methods do not allow to specify strides. If you need to specify strides, you have to
      * use the Map class directly.
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

    using Base::setConstant;
    Derived& setConstant(int size, const Scalar& value);
    Derived& setConstant(int rows, int cols, const Scalar& value);

    using Base::setZero;
    Derived& setZero(int size);
    Derived& setZero(int rows, int cols);

    using Base::setOnes;
    Derived& setOnes(int size);
    Derived& setOnes(int rows, int cols);

    using Base::setRandom;
    Derived& setRandom(int size);
    Derived& setRandom(int rows, int cols);

    #ifdef EIGEN_DENSESTORAGEBASE_PLUGIN
    #include EIGEN_DENSESTORAGEBASE_PLUGIN
    #endif

  protected:
    /** \internal Resizes *this in preparation for assigning \a other to it.
      * Takes care of doing all the checking that's needed.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE void _resize_to_match(const DenseBase<OtherDerived>& other)
    {
      #ifdef EIGEN_NO_AUTOMATIC_RESIZING
      ei_assert((this->size()==0 || (IsVectorAtCompileTime ? (this->size() == other.size())
                 : (rows() == other.rows() && cols() == other.cols())))
        && "Size mismatch. Automatic resizing is disabled because EIGEN_NO_AUTOMATIC_RESIZING is defined");
      #endif
      resizeLike(other);
    }

    /**
      * \brief Copies the value of the expression \a other into \c *this with automatic resizing.
      *
      * *this might be resized to match the dimensions of \a other. If *this was a null matrix (not already initialized),
      * it will be initialized.
      *
      * Note that copying a row-vector into a vector (and conversely) is allowed.
      * The resizing, if any, is then done in the appropriate way so that row-vectors
      * remain row-vectors and vectors remain vectors.
      *
      * \sa operator=(const MatrixBase<OtherDerived>&), _set_noalias()
      *
      * \internal
      */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Derived& _set(const DenseBase<OtherDerived>& other)
    {
      _set_selector(other.derived(), typename ei_meta_if<static_cast<bool>(int(OtherDerived::Flags) & EvalBeforeAssigningBit), ei_meta_true, ei_meta_false>::ret());
      return this->derived();
    }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE void _set_selector(const OtherDerived& other, const ei_meta_true&) { _set_noalias(other.eval()); }

    template<typename OtherDerived>
    EIGEN_STRONG_INLINE void _set_selector(const OtherDerived& other, const ei_meta_false&) { _set_noalias(other); }

    /** \internal Like _set() but additionally makes the assumption that no aliasing effect can happen (which
      * is the case when creating a new matrix) so one can enforce lazy evaluation.
      *
      * \sa operator=(const MatrixBase<OtherDerived>&), _set()
      */
    template<typename OtherDerived>
    EIGEN_STRONG_INLINE Derived& _set_noalias(const DenseBase<OtherDerived>& other)
    {
      _resize_to_match(other);
      // the 'false' below means to enforce lazy evaluation. We don't use lazyAssign() because
      // it wouldn't allow to copy a row-vector into a column-vector.
      return ei_assign_selector<Derived,OtherDerived,false>::run(this->derived(), other.derived());
    }

    EIGEN_STRONG_INLINE void _check_template_params()
    {
      EIGEN_STATIC_ASSERT(((RowsAtCompileTime >= MaxRowsAtCompileTime)
                        && (ColsAtCompileTime >= MaxColsAtCompileTime)
                        && (MaxRowsAtCompileTime >= 0)
                        && (MaxColsAtCompileTime >= 0)
                        && (RowsAtCompileTime <= Dynamic)
                        && (ColsAtCompileTime <= Dynamic)
                        && (MaxRowsAtCompileTime == RowsAtCompileTime || RowsAtCompileTime==Dynamic)
                        && (MaxColsAtCompileTime == ColsAtCompileTime || ColsAtCompileTime==Dynamic)
                        && ((MaxRowsAtCompileTime==Dynamic?1:MaxRowsAtCompileTime)*(MaxColsAtCompileTime==Dynamic?1:MaxColsAtCompileTime)<Dynamic)
                        && (_Options & (DontAlign|RowMajor)) == _Options),
        INVALID_MATRIX_TEMPLATE_PARAMETERS)
    }


    template<typename T0, typename T1>
    EIGEN_STRONG_INLINE void _init2(int rows, int cols, typename ei_enable_if<Base::SizeAtCompileTime!=2,T0>::type* = 0)
    {
      ei_assert(rows > 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
             && cols > 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
      m_storage.resize(rows*cols,rows,cols);
      EIGEN_INITIALIZE_BY_ZERO_IF_THAT_OPTION_IS_ENABLED
    }
    template<typename T0, typename T1>
    EIGEN_STRONG_INLINE void _init2(const Scalar& x, const Scalar& y, typename ei_enable_if<Base::SizeAtCompileTime==2,T0>::type* = 0)
    {
      EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DenseStorageBase, 2)
      m_storage.data()[0] = x;
      m_storage.data()[1] = y;
    }

    template<typename MatrixTypeA, typename MatrixTypeB, bool SwapPointers>
    friend struct ei_matrix_swap_impl;

    /** \internal generic implementation of swap for dense storage since for dynamic-sized matrices of same type it is enough to swap the
      * data pointers.
      */
    template<typename OtherDerived>
    void _swap(DenseBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other)
    {
      enum { SwapPointers = ei_is_same_type<Derived, OtherDerived>::ret && Base::SizeAtCompileTime==Dynamic };
      ei_matrix_swap_impl<Derived, OtherDerived, bool(SwapPointers)>::run(this->derived(), other.const_cast_derived());
    }
};



template <typename Derived, typename OtherDerived, bool IsVector>
struct ei_conservative_resize_like_impl
{
  static void run(DenseBase<Derived>& _this, int rows, int cols)
  {
    if (_this.rows() == rows && _this.cols() == cols) return;
    EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(Derived)

    if ( ( Derived::IsRowMajor && _this.cols() == cols) || // row-major and we change only the number of rows
         (!Derived::IsRowMajor && _this.rows() == rows) )  // column-major and we change only the number of columns
    {
      _this.derived().m_storage.conservativeResize(rows*cols,rows,cols);
    }
    else
    {
      // The storage order does not allow us to use reallocation.
      typename Derived::PlainObject tmp(rows,cols);
      const int common_rows = std::min(rows, _this.rows());
      const int common_cols = std::min(cols, _this.cols());
      tmp.block(0,0,common_rows,common_cols) = _this.block(0,0,common_rows,common_cols);
      _this.derived().swap(tmp);
    }
  }

  static void run(DenseBase<Derived>& _this, const DenseBase<OtherDerived>& other)
  {
    if (_this.rows() == other.rows() && _this.cols() == other.cols()) return;

    // Note: Here is space for improvement. Basically, for conservativeResize(int,int),
    // neither RowsAtCompileTime or ColsAtCompileTime must be Dynamic. If only one of the
    // dimensions is dynamic, one could use either conservativeResize(int rows, NoChange_t) or
    // conservativeResize(NoChange_t, int cols). For these methods new static asserts like
    // EIGEN_STATIC_ASSERT_DYNAMIC_ROWS and EIGEN_STATIC_ASSERT_DYNAMIC_COLS would be good.
    EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(Derived)
    EIGEN_STATIC_ASSERT_DYNAMIC_SIZE(OtherDerived)

    if ( ( Derived::IsRowMajor && _this.cols() == other.cols()) || // row-major and we change only the number of rows
         (!Derived::IsRowMajor && _this.rows() == other.rows()) )  // column-major and we change only the number of columns
    {
      const int new_rows = other.rows() - _this.rows();
      const int new_cols = other.cols() - _this.cols();
      _this.derived().m_storage.conservativeResize(other.size(),other.rows(),other.cols());
      if (new_rows>0)
        _this.corner(BottomRight, new_rows, other.cols()) = other.corner(BottomRight, new_rows, other.cols());
      else if (new_cols>0)
        _this.corner(BottomRight, other.rows(), new_cols) = other.corner(BottomRight, other.rows(), new_cols);
    }
    else
    {
      // The storage order does not allow us to use reallocation.
      typename Derived::PlainObject tmp(other);
      const int common_rows = std::min(tmp.rows(), _this.rows());
      const int common_cols = std::min(tmp.cols(), _this.cols());
      tmp.block(0,0,common_rows,common_cols) = _this.block(0,0,common_rows,common_cols);
      _this.derived().swap(tmp);
    }
  }
};

template <typename Derived, typename OtherDerived>
struct ei_conservative_resize_like_impl<Derived,OtherDerived,true>
{
  static void run(DenseBase<Derived>& _this, int size)
  {
    const int new_rows = Derived::RowsAtCompileTime==1 ? 1 : size;
    const int new_cols = Derived::RowsAtCompileTime==1 ? size : 1;
    _this.derived().m_storage.conservativeResize(size,new_rows,new_cols);
  }

  static void run(DenseBase<Derived>& _this, const DenseBase<OtherDerived>& other)
  {
    if (_this.rows() == other.rows() && _this.cols() == other.cols()) return;

    const int num_new_elements = other.size() - _this.size();

    const int new_rows = Derived::RowsAtCompileTime==1 ? 1 : other.rows();
    const int new_cols = Derived::RowsAtCompileTime==1 ? other.cols() : 1;
    _this.derived().m_storage.conservativeResize(other.size(),new_rows,new_cols);

    if (num_new_elements > 0)
      _this.tail(num_new_elements) = other.tail(num_new_elements);
  }
};

template<typename MatrixTypeA, typename MatrixTypeB, bool SwapPointers>
struct ei_matrix_swap_impl
{
  static inline void run(MatrixTypeA& a, MatrixTypeB& b)
  {
    a.base().swap(b);
  }
};

template<typename MatrixTypeA, typename MatrixTypeB>
struct ei_matrix_swap_impl<MatrixTypeA, MatrixTypeB, true>
{
  static inline void run(MatrixTypeA& a, MatrixTypeB& b)
  {
    static_cast<typename MatrixTypeA::Base&>(a).m_storage.swap(static_cast<typename MatrixTypeB::Base&>(b).m_storage);
  }
};

#endif // EIGEN_DENSESTORAGEBASE_H
