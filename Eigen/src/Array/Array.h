// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_ARRAYBASE_H
#define EIGEN_ARRAYBASE_H

/** \ingroup Array_Module
  *
  * \class ArrayBase
  *
  * \brief Base class for all 1D and 2D array, and related expressions
  *
  * An array is similar to a dense vector or matrix. While matrices are mathematical
  * objects with well defined linear algebra operators, an array is just a collection
  * of scalar values arranged in a one or two dimensionnal fashion. The main consequence,
  * is that all operations applied to an array are performed coefficient wise. Furthermore,
  * arays support scalar math functions of the c++ standard library, and convenient
  * constructors allowing to easily write generic code working for both scalar values
  * and arrays.
  *
  * This class is the base that is inherited by all array expression types.
  *
  * \param Derived is the derived type, e.g. an array type, or an expression, etc.
  *
  * \sa class ArrayBase
  */
template<typename Derived> class ArrayBase
#ifndef EIGEN_PARSED_BY_DOXYGEN
  : public ei_special_scalar_op_base<Derived,typename ei_traits<Derived>::Scalar,
                                     typename NumTraits<typename ei_traits<Derived>::Scalar>::Real>
#endif // not EIGEN_PARSED_BY_DOXYGEN
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** The base class for a given storage type. */
    typedef ArrayBase StorageBaseType;
    /** Construct the base class type for the derived class OtherDerived */
    template <typename OtherDerived> struct MakeBase { typedef ArrayBase<OtherDerived> Type; };

    using ei_special_scalar_op_base<Derived,typename ei_traits<Derived>::Scalar,
                typename NumTraits<typename ei_traits<Derived>::Scalar>::Real>::operator*;

    class InnerIterator;

    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename ei_packet_traits<Scalar>::type PacketScalar;
#endif // not EIGEN_PARSED_BY_DOXYGEN

    // FIXME A lot of this stuff could be moved to AnyArrayBase, I guess

    enum {

      RowsAtCompileTime = ei_traits<Derived>::RowsAtCompileTime,
        /**< The number of rows at compile-time. This is just a copy of the value provided
          * by the \a Derived type. If a value is not known at compile-time,
          * it is set to the \a Dynamic constant.
          * \sa ArrayBase::rows(), ArrayBase::cols(), ColsAtCompileTime, SizeAtCompileTime */

      ColsAtCompileTime = ei_traits<Derived>::ColsAtCompileTime,
        /**< The number of columns at compile-time. This is just a copy of the value provided
          * by the \a Derived type. If a value is not known at compile-time,
          * it is set to the \a Dynamic constant.
          * \sa ArrayBase::rows(), ArrayBase::cols(), RowsAtCompileTime, SizeAtCompileTime */


      SizeAtCompileTime = (ei_size_at_compile_time<ei_traits<Derived>::RowsAtCompileTime,
                                                   ei_traits<Derived>::ColsAtCompileTime>::ret),
        /**< This is equal to the number of coefficients, i.e. the number of
          * rows times the number of columns, or to \a Dynamic if this is not
          * known at compile-time. \sa RowsAtCompileTime, ColsAtCompileTime */

      MaxRowsAtCompileTime = ei_traits<Derived>::MaxRowsAtCompileTime,
        /**< This value is equal to the maximum possible number of rows that this expression
          * might have. If this expression might have an arbitrarily high number of rows,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa RowsAtCompileTime, MaxColsAtCompileTime, MaxSizeAtCompileTime
          */

      MaxColsAtCompileTime = ei_traits<Derived>::MaxColsAtCompileTime,
        /**< This value is equal to the maximum possible number of columns that this expression
          * might have. If this expression might have an arbitrarily high number of columns,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa ColsAtCompileTime, MaxRowsAtCompileTime, MaxSizeAtCompileTime
          */

      MaxSizeAtCompileTime = (ei_size_at_compile_time<ei_traits<Derived>::MaxRowsAtCompileTime,
                                                      ei_traits<Derived>::MaxColsAtCompileTime>::ret),
        /**< This value is equal to the maximum possible number of coefficients that this expression
          * might have. If this expression might have an arbitrarily high number of coefficients,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa SizeAtCompileTime, MaxRowsAtCompileTime, MaxColsAtCompileTime
          */

      IsVectorAtCompileTime = ei_traits<Derived>::RowsAtCompileTime == 1
                           || ei_traits<Derived>::ColsAtCompileTime == 1,
        /**< This is set to true if either the number of rows or the number of
          * columns is known at compile-time to be equal to 1. Indeed, in that case,
          * we are dealing with a column-vector (if there is only one column) or with
          * a row-vector (if there is only one row). */

      Flags = ei_traits<Derived>::Flags,
        /**< This stores expression \ref flags flags which may or may not be inherited by new expressions
          * constructed from this one. See the \ref flags "list of flags".
          */

      CoeffReadCost = ei_traits<Derived>::CoeffReadCost,
        /**< This is a rough measure of how expensive it is to read one coefficient from
          * this expression.
          */

#ifndef EIGEN_PARSED_BY_DOXYGEN
      _HasDirectAccess = (int(Flags)&DirectAccessBit) ? 1 : 0 // workaround sunCC
#endif
    };

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** This is the "real scalar" type; if the \a Scalar type is already real numbers
      * (e.g. int, float or double) then \a RealScalar is just the same as \a Scalar. If
      * \a Scalar is \a std::complex<T> then RealScalar is \a T.
      *
      * \sa class NumTraits
      */
    typedef typename NumTraits<Scalar>::Real RealScalar;

    /** type of the equivalent square matrix */
    typedef Matrix<Scalar,EIGEN_ENUM_MAX(RowsAtCompileTime,ColsAtCompileTime),
                          EIGEN_ENUM_MAX(RowsAtCompileTime,ColsAtCompileTime)> SquareMatrixType;
#endif // not EIGEN_PARSED_BY_DOXYGEN

    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    inline int rows() const { return derived().rows(); }
    /** \returns the number of columns. \sa rows(), ColsAtCompileTime*/
    inline int cols() const { return derived().cols(); }
    /** \returns the number of coefficients, which is rows()*cols().
      * \sa rows(), cols(), SizeAtCompileTime. */
    inline int size() const { return rows() * cols(); }
    /** \returns the number of nonzero coefficients which is in practice the number
      * of stored coefficients. */
    inline int nonZeros() const { return size(); }
    /** \returns true if either the number of rows or the number of columns is equal to 1.
      * In other words, this function returns
      * \code rows()==1 || cols()==1 \endcode
      * \sa rows(), cols(), IsVectorAtCompileTime. */
    inline bool isVector() const { return rows()==1 || cols()==1; }
    /** \returns the size of the storage major dimension,
      * i.e., the number of columns for a columns major matrix, and the number of rows otherwise */
    int outerSize() const { return (int(Flags)&RowMajorBit) ? this->rows() : this->cols(); }
    /** \returns the size of the inner dimension according to the storage order,
      * i.e., the number of rows for a columns major matrix, and the number of cols otherwise */
    int innerSize() const { return (int(Flags)&RowMajorBit) ? this->cols() : this->rows(); }

    /** Only plain matrices, not expressions may be resized; therefore the only useful resize method is
      * Matrix::resize(). The present method only asserts that the new size equals the old size, and does
      * nothing else.
      */
    void resize(int size)
    {
      ei_assert(size == this->size()
                && "ArrayBase::resize() does not actually allow to resize.");
    }
    /** Only plain matrices, not expressions may be resized; therefore the only useful resize method is
      * Matrix::resize(). The present method only asserts that the new size equals the old size, and does
      * nothing else.
      */
    void resize(int rows, int cols)
    {
      ei_assert(rows == this->rows() && cols == this->cols()
                && "ArrayBase::resize() does not actually allow to resize.");
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal the plain matrix type corresponding to this expression. Note that is not necessarily
      * exactly the return type of eval(): in the case of plain matrices, the return type of eval() is a const
      * reference to a matrix, not a matrix! It is however guaranteed that the return type of eval() is either
      * PlainMatrixType or const PlainMatrixType&.
      */
    typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType;
    /** \internal the column-major plain matrix type corresponding to this expression. Note that is not necessarily
      * exactly the return type of eval(): in the case of plain matrices, the return type of eval() is a const
      * reference to a matrix, not a matrix!
      * The only difference from PlainMatrixType is that PlainMatrixType_ColMajor is guaranteed to be column-major.
      */
    typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType_ColMajor;

    /** \internal the return type of coeff()
      */
    typedef typename ei_meta_if<_HasDirectAccess, const Scalar&, Scalar>::ret CoeffReturnType;

    /** \internal Represents a matrix with all coefficients equal to one another*/
    typedef CwiseNullaryOp<ei_scalar_constant_op<Scalar>,Derived> ConstantReturnType;
    /** \internal expression tyepe of a column */
    typedef Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1> ColXpr;
    /** \internal expression tyepe of a column */
    typedef Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime> RowXpr;
#endif // not EIGEN_PARSED_BY_DOXYGEN

    #define EIGEN_CURRENT_STORAGE_BASE_CLASS Eigen::ArrayBase
    #include "../Core/CommonCwiseUnaryOps.h"
    #include "ArrayCwiseUnaryOps.h"
    #include "../Core/CommonCwiseBinaryOps.h"
    #include "ArrayCwiseBinaryOps.h"
    #undef EIGEN_CURRENT_STORAGE_BASE_CLASS

    /** Copies \a other into *this. \returns a reference to *this. */
    template<typename OtherDerived>
    Derived& operator=(const ArrayBase<OtherDerived>& other);

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    Derived& operator=(const ArrayBase& other);

    template<typename OtherDerived>
    Derived& operator=(const AnyArrayBase<OtherDerived> &other);

    template<typename OtherDerived>
    Derived& operator+=(const AnyArrayBase<OtherDerived> &other);

    template<typename OtherDerived>
    Derived& operator-=(const AnyArrayBase<OtherDerived> &other);

    template<typename OtherDerived>
    Derived& operator=(const ReturnByValue<OtherDerived>& func);

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Copies \a other into *this without evaluating other. \returns a reference to *this. */
    template<typename OtherDerived>
    Derived& lazyAssign(const ArrayBase<OtherDerived>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    CommaInitializer<Derived> operator<< (const Scalar& s);

    template<typename OtherDerived>
    CommaInitializer<Derived> operator<< (const ArrayBase<OtherDerived>& other);

    const CoeffReturnType coeff(int row, int col) const;
    const CoeffReturnType operator()(int row, int col) const;

    Scalar& coeffRef(int row, int col);
    Scalar& operator()(int row, int col);

    const CoeffReturnType coeff(int index) const;
    const CoeffReturnType operator[](int index) const;
    const CoeffReturnType operator()(int index) const;

    Scalar& coeffRef(int index);
    Scalar& operator[](int index);
    Scalar& operator()(int index);

#ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename OtherDerived>
    void copyCoeff(int row, int col, const ArrayBase<OtherDerived>& other);
    template<typename OtherDerived>
    void copyCoeff(int index, const ArrayBase<OtherDerived>& other);
    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(int row, int col, const ArrayBase<OtherDerived>& other);
    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(int index, const ArrayBase<OtherDerived>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    template<int LoadMode>
    PacketScalar packet(int row, int col) const;
    template<int StoreMode>
    void writePacket(int row, int col, const PacketScalar& x);

    template<int LoadMode>
    PacketScalar packet(int index) const;
    template<int StoreMode>
    void writePacket(int index, const PacketScalar& x);

    template<typename OtherDerived>
    Derived& operator+=(const ArrayBase<OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const ArrayBase<OtherDerived>& other);

    template<typename OtherDerived>
    Derived& operator*=(const ArrayBase<OtherDerived>& other);

    Eigen::Transpose<Derived> transpose();
    const Eigen::Transpose<Derived> transpose() const;
    void transposeInPlace();

    #ifndef EIGEN_NO_DEBUG
    template<typename OtherDerived>
    Derived& lazyAssign(const Transpose<OtherDerived>& other);
    template<typename DerivedA, typename DerivedB>
    Derived& lazyAssign(const CwiseBinaryOp<ei_scalar_sum_op<Scalar>,Transpose<DerivedA>,DerivedB>& other);
    template<typename DerivedA, typename DerivedB>
    Derived& lazyAssign(const CwiseBinaryOp<ei_scalar_sum_op<Scalar>,DerivedA,Transpose<DerivedB> >& other);

    template<typename OtherDerived>
    Derived& lazyAssign(const CwiseUnaryOp<ei_scalar_conjugate_op<Scalar>, NestByValue<Eigen::Transpose<OtherDerived> > >& other);
    template<typename DerivedA, typename DerivedB>
    Derived& lazyAssign(const CwiseBinaryOp<ei_scalar_sum_op<Scalar>,CwiseUnaryOp<ei_scalar_conjugate_op<Scalar>, NestByValue<Eigen::Transpose<DerivedA> > >,DerivedB>& other);
    template<typename DerivedA, typename DerivedB>
    Derived& lazyAssign(const CwiseBinaryOp<ei_scalar_sum_op<Scalar>,DerivedA,CwiseUnaryOp<ei_scalar_conjugate_op<Scalar>, NestByValue<Eigen::Transpose<DerivedB> > > >& other);
    #endif

    RowXpr row(int i);
    const RowXpr row(int i) const;

    ColXpr col(int i);
    const ColXpr col(int i) const;

    Minor<Derived> minor(int row, int col);
    const Minor<Derived> minor(int row, int col) const;

    typename BlockReturnType<Derived>::Type block(int startRow, int startCol, int blockRows, int blockCols);
    const typename BlockReturnType<Derived>::Type
    block(int startRow, int startCol, int blockRows, int blockCols) const;

    VectorBlock<Derived> segment(int start, int size);
    const VectorBlock<Derived> segment(int start, int size) const;

    VectorBlock<Derived> start(int size);
    const VectorBlock<Derived> start(int size) const;

    VectorBlock<Derived> end(int size);
    const VectorBlock<Derived> end(int size) const;

    typename BlockReturnType<Derived>::Type corner(CornerType type, int cRows, int cCols);
    const typename BlockReturnType<Derived>::Type corner(CornerType type, int cRows, int cCols) const;

    template<int BlockRows, int BlockCols>
    typename BlockReturnType<Derived, BlockRows, BlockCols>::Type block(int startRow, int startCol);
    template<int BlockRows, int BlockCols>
    const typename BlockReturnType<Derived, BlockRows, BlockCols>::Type block(int startRow, int startCol) const;

    template<int CRows, int CCols>
    typename BlockReturnType<Derived, CRows, CCols>::Type corner(CornerType type);
    template<int CRows, int CCols>
    const typename BlockReturnType<Derived, CRows, CCols>::Type corner(CornerType type) const;

    template<int Size> VectorBlock<Derived,Size> start(void);
    template<int Size> const VectorBlock<Derived,Size> start() const;

    template<int Size> VectorBlock<Derived,Size> end();
    template<int Size> const VectorBlock<Derived,Size> end() const;

    template<int Size> VectorBlock<Derived,Size> segment(int start);
    template<int Size> const VectorBlock<Derived,Size> segment(int start) const;

    static const ConstantReturnType
    Constant(int rows, int cols, const Scalar& value);
    static const ConstantReturnType
    Constant(int size, const Scalar& value);
    static const ConstantReturnType
    Constant(const Scalar& value);

    template<typename CustomNullaryOp>
    static const CwiseNullaryOp<CustomNullaryOp, Derived>
    NullaryExpr(int rows, int cols, const CustomNullaryOp& func);
    template<typename CustomNullaryOp>
    static const CwiseNullaryOp<CustomNullaryOp, Derived>
    NullaryExpr(int size, const CustomNullaryOp& func);
    template<typename CustomNullaryOp>
    static const CwiseNullaryOp<CustomNullaryOp, Derived>
    NullaryExpr(const CustomNullaryOp& func);

    static const ConstantReturnType Zero(int rows, int cols);
    static const ConstantReturnType Zero(int size);
    static const ConstantReturnType Zero();
    static const ConstantReturnType Ones(int rows, int cols);
    static const ConstantReturnType Ones(int size);
    static const ConstantReturnType Ones();

    void fill(const Scalar& value);
    Derived& setConstant(const Scalar& value);
    Derived& setZero();
    Derived& setOnes();
    Derived& setRandom();


    template<typename OtherDerived>
    bool isApprox(const ArrayBase<OtherDerived>& other,
                  RealScalar prec = precision<Scalar>()) const;
    bool isMuchSmallerThan(const RealScalar& other,
                           RealScalar prec = precision<Scalar>()) const;
    template<typename OtherDerived>
    bool isMuchSmallerThan(const ArrayBase<OtherDerived>& other,
                           RealScalar prec = precision<Scalar>()) const;

    bool isApproxToConstant(const Scalar& value, RealScalar prec = precision<Scalar>()) const;
    bool isConstant(const Scalar& value, RealScalar prec = precision<Scalar>()) const;
    bool isZero(RealScalar prec = precision<Scalar>()) const;
    bool isOnes(RealScalar prec = precision<Scalar>()) const;
    bool isIdentity(RealScalar prec = precision<Scalar>()) const;
    bool isDiagonal(RealScalar prec = precision<Scalar>()) const;

    bool isUpperTriangular(RealScalar prec = precision<Scalar>()) const;
    bool isLowerTriangular(RealScalar prec = precision<Scalar>()) const;

    template<typename OtherDerived>
    bool isOrthogonal(const ArrayBase<OtherDerived>& other,
                      RealScalar prec = precision<Scalar>()) const;
    bool isUnitary(RealScalar prec = precision<Scalar>()) const;

    template<typename OtherDerived>
    inline bool operator==(const ArrayBase<OtherDerived>& other) const
    { return cwiseEqual(other).all(); }

    template<typename OtherDerived>
    inline bool operator!=(const ArrayBase<OtherDerived>& other) const
    { return cwiseNotEqual(other).all(); }


    /** \returns the matrix or vector obtained by evaluating this expression.
      *
      * Notice that in the case of a plain matrix or vector (not an expression) this function just returns
      * a const reference, in order to avoid a useless copy.
      */
    EIGEN_STRONG_INLINE const typename ei_eval<Derived>::type eval() const
    { return typename ei_eval<Derived>::type(derived()); }

    template<typename OtherDerived>
    void swap(ArrayBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other);

    NoAlias<Derived,Eigen::ArrayBase > noalias();

    /** \returns number of elements to skip to pass from one row (resp. column) to another
      * for a row-major (resp. column-major) matrix.
      * Combined with coeffRef() and the \ref flags flags, it allows a direct access to the data
      * of the underlying matrix.
      */
    inline int stride(void) const { return derived().stride(); }

    inline const NestByValue<Derived> nestByValue() const;

    Scalar sum() const;
    Scalar mean() const;
    Scalar trace() const;

    Scalar prod() const;

    typename ei_traits<Derived>::Scalar minCoeff() const;
    typename ei_traits<Derived>::Scalar maxCoeff() const;

    typename ei_traits<Derived>::Scalar minCoeff(int* row, int* col) const;
    typename ei_traits<Derived>::Scalar maxCoeff(int* row, int* col) const;

    typename ei_traits<Derived>::Scalar minCoeff(int* index) const;
    typename ei_traits<Derived>::Scalar maxCoeff(int* index) const;

    template<typename BinaryOp>
    typename ei_result_of<BinaryOp(typename ei_traits<Derived>::Scalar)>::type
    redux(const BinaryOp& func) const;

    template<typename Visitor>
    void visit(Visitor& func) const;

#ifndef EIGEN_PARSED_BY_DOXYGEN
    using AnyArrayBase<Derived>::derived;
    inline Derived& const_cast_derived() const
    { return *static_cast<Derived*>(const_cast<ArrayBase*>(this)); }
#endif // not EIGEN_PARSED_BY_DOXYGEN

    inline const WithFormat<Derived> format(const IOFormat& fmt) const;

    bool all(void) const;
    bool any(void) const;
    int count() const;

    const VectorwiseOp<Derived,Horizontal> rowwise() const;
    VectorwiseOp<Derived,Horizontal> rowwise();
    const VectorwiseOp<Derived,Vertical> colwise() const;
    VectorwiseOp<Derived,Vertical> colwise();

    static const CwiseNullaryOp<ei_scalar_random_op<Scalar>,Derived> Random(int rows, int cols);
    static const CwiseNullaryOp<ei_scalar_random_op<Scalar>,Derived> Random(int size);
    static const CwiseNullaryOp<ei_scalar_random_op<Scalar>,Derived> Random();

    template<typename ThenDerived,typename ElseDerived>
    const Select<Derived,ThenDerived,ElseDerived>
    select(const ArrayBase<ThenDerived>& thenMatrix,
           const ArrayBase<ElseDerived>& elseMatrix) const;

    template<typename ThenDerived>
    inline const Select<Derived,ThenDerived, NestByValue<typename ThenDerived::ConstantReturnType> >
    select(const ArrayBase<ThenDerived>& thenMatrix, typename ThenDerived::Scalar elseScalar) const;

    template<typename ElseDerived>
    inline const Select<Derived, NestByValue<typename ElseDerived::ConstantReturnType>, ElseDerived >
    select(typename ElseDerived::Scalar thenScalar, const ArrayBase<ElseDerived>& elseMatrix) const;

    template<int RowFactor, int ColFactor>
    const Replicate<Derived,RowFactor,ColFactor> replicate() const;
    const Replicate<Derived,Dynamic,Dynamic> replicate(int rowFacor,int colFactor) const;

    Eigen::Reverse<Derived, BothDirections> reverse();
    const Eigen::Reverse<Derived, BothDirections> reverse() const;
    void reverseInPlace();

    #ifdef EIGEN_MATRIXBASE_PLUGIN
    #include EIGEN_MATRIXBASE_PLUGIN
    #endif

  protected:
    /** Default constructor. Do nothing. */
    ArrayBase()
    {
      /* Just checks for self-consistency of the flags.
       * Only do it when debugging Eigen, as this borders on paranoiac and could slow compilation down
       */
#ifdef EIGEN_INTERNAL_DEBUGGING
      EIGEN_STATIC_ASSERT(ei_are_flags_consistent<Flags>::ret,
                          INVALID_MATRIXBASE_TEMPLATE_PARAMETERS)
#endif
    }

  private:
    explicit ArrayBase(int);
    ArrayBase(int,int);
    template<typename OtherDerived> explicit ArrayBase(const ArrayBase<OtherDerived>&);
};

#endif // EIGEN_ARRAYBASE_H
