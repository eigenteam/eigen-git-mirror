// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_DENSEBASE_H
#define EIGEN_DENSEBASE_H

/** \class DenseBase
  *
  * \brief Base class for all dense matrices, vectors, and arrays
  *
  * This class is the base that is inherited by all dense objects (matrix, vector, arrays,
  * and related expression types). The common Eigen API for dense objects is contained in this class.
  *
  * \param Derived is the derived type, e.g., a matrix type or an expression.
  */
template<typename Derived> class DenseBase
#ifndef EIGEN_PARSED_BY_DOXYGEN
  : public ei_special_scalar_op_base<Derived,typename ei_traits<Derived>::Scalar,
                                     typename NumTraits<typename ei_traits<Derived>::Scalar>::Real>
#else
  : public AnyMatrixBase<Derived>
#endif // not EIGEN_PARSED_BY_DOXYGEN
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    using ei_special_scalar_op_base<Derived,typename ei_traits<Derived>::Scalar,
                typename NumTraits<typename ei_traits<Derived>::Scalar>::Real>::operator*;

    class InnerIterator;

    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename ei_packet_traits<Scalar>::type PacketScalar;

    using AnyMatrixBase<Derived>::derived;
    using AnyMatrixBase<Derived>::const_cast_derived;
#endif // not EIGEN_PARSED_BY_DOXYGEN

    enum {

      RowsAtCompileTime = ei_traits<Derived>::RowsAtCompileTime,
        /**< The number of rows at compile-time. This is just a copy of the value provided
          * by the \a Derived type. If a value is not known at compile-time,
          * it is set to the \a Dynamic constant.
          * \sa MatrixBase::rows(), MatrixBase::cols(), ColsAtCompileTime, SizeAtCompileTime */

      ColsAtCompileTime = ei_traits<Derived>::ColsAtCompileTime,
        /**< The number of columns at compile-time. This is just a copy of the value provided
          * by the \a Derived type. If a value is not known at compile-time,
          * it is set to the \a Dynamic constant.
          * \sa MatrixBase::rows(), MatrixBase::cols(), RowsAtCompileTime, SizeAtCompileTime */


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
                && "MatrixBase::resize() does not actually allow to resize.");
    }
    /** Only plain matrices, not expressions may be resized; therefore the only useful resize method is
      * Matrix::resize(). The present method only asserts that the new size equals the old size, and does
      * nothing else.
      */
    void resize(int rows, int cols)
    {
      ei_assert(rows == this->rows() && cols == this->cols()
                && "MatrixBase::resize() does not actually allow to resize.");
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal the return type of coeff()
      */
    typedef typename ei_meta_if<_HasDirectAccess, const Scalar&, Scalar>::ret CoeffReturnType;

    /** \internal Represents a matrix with all coefficients equal to one another*/
    typedef CwiseNullaryOp<ei_scalar_constant_op<Scalar>,Derived> ConstantReturnType;
    /** \internal Represents a vector with linearly spaced coefficients that allows sequential access only. */
    typedef CwiseNullaryOp<ei_linspaced_op<Scalar,false>,Derived> SequentialLinSpacedReturnType;
    /** \internal Represents a vector with linearly spaced coefficients that allows random access. */
    typedef CwiseNullaryOp<ei_linspaced_op<Scalar,true>,Derived> RandomAccessLinSpacedReturnType;
    /** \internal the return type of MatrixBase::eigenvalues() */
    typedef Matrix<typename NumTraits<typename ei_traits<Derived>::Scalar>::Real, ei_traits<Derived>::ColsAtCompileTime, 1> EigenvaluesReturnType;
    /** \internal expression type of a column */
    typedef Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1> ColXpr;
    /** \internal expression type of a column */
    typedef Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime> RowXpr;
#endif // not EIGEN_PARSED_BY_DOXYGEN

    /** Copies \a other into *this. \returns a reference to *this. */
    template<typename OtherDerived>
    Derived& operator=(const DenseBase<OtherDerived>& other);

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    Derived& operator=(const DenseBase& other);

    template<typename OtherDerived>
    Derived& operator=(const AnyMatrixBase<OtherDerived> &other);

    template<typename OtherDerived>
    Derived& operator+=(const AnyMatrixBase<OtherDerived> &other);

    template<typename OtherDerived>
    Derived& operator-=(const AnyMatrixBase<OtherDerived> &other);

    template<typename OtherDerived>
    Derived& operator=(const ReturnByValue<OtherDerived>& func);

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Copies \a other into *this without evaluating other. \returns a reference to *this. */
    template<typename OtherDerived>
    Derived& lazyAssign(const DenseBase<OtherDerived>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    CommaInitializer<Derived> operator<< (const Scalar& s);

    template<unsigned int Added,unsigned int Removed>
    const Flagged<Derived, Added, Removed> flagged() const;

    template<typename OtherDerived>
    CommaInitializer<Derived> operator<< (const DenseBase<OtherDerived>& other);

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
    void copyCoeff(int row, int col, const DenseBase<OtherDerived>& other);
    template<typename OtherDerived>
    void copyCoeff(int index, const DenseBase<OtherDerived>& other);
    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(int row, int col, const DenseBase<OtherDerived>& other);
    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(int index, const DenseBase<OtherDerived>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    template<int LoadMode>
    PacketScalar packet(int row, int col) const;
    template<int StoreMode>
    void writePacket(int row, int col, const PacketScalar& x);

    template<int LoadMode>
    PacketScalar packet(int index) const;
    template<int StoreMode>
    void writePacket(int index, const PacketScalar& x);

    Eigen::Transpose<Derived> transpose();
    const Eigen::Transpose<Derived> transpose() const;
    void transposeInPlace();
#ifndef EIGEN_NO_DEBUG
  protected:
    template<typename OtherDerived>
    void checkTransposeAliasing(const OtherDerived& other) const;
  public:
#endif

    RowXpr row(int i);
    const RowXpr row(int i) const;

    ColXpr col(int i);
    const ColXpr col(int i) const;

    typename BlockReturnType<Derived>::Type block(int startRow, int startCol, int blockRows, int blockCols);
    const typename BlockReturnType<Derived>::Type
    block(int startRow, int startCol, int blockRows, int blockCols) const;

    VectorBlock<Derived> segment(int start, int size);
    const VectorBlock<Derived> segment(int start, int size) const;

    VectorBlock<Derived> head(int size);
    const VectorBlock<Derived> head(int size) const;

    VectorBlock<Derived> tail(int size);
    const VectorBlock<Derived> tail(int size) const;

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

    template<int Size> VectorBlock<Derived,Size> head(void);
    template<int Size> const VectorBlock<Derived,Size> head() const;

    template<int Size> VectorBlock<Derived,Size> tail();
    template<int Size> const VectorBlock<Derived,Size> tail() const;

    template<int Size> VectorBlock<Derived,Size> segment(int start);
    template<int Size> const VectorBlock<Derived,Size> segment(int start) const;

    Diagonal<Derived,0> diagonal();
    const Diagonal<Derived,0> diagonal() const;

    template<int Index> Diagonal<Derived,Index> diagonal();
    template<int Index> const Diagonal<Derived,Index> diagonal() const;

    Diagonal<Derived, Dynamic> diagonal(int index);
    const Diagonal<Derived, Dynamic> diagonal(int index) const;

    template<unsigned int Mode> TriangularView<Derived, Mode> part();
    template<unsigned int Mode> const TriangularView<Derived, Mode> part() const;

    template<unsigned int Mode> TriangularView<Derived, Mode> triangularView();
    template<unsigned int Mode> const TriangularView<Derived, Mode> triangularView() const;

    template<unsigned int UpLo> SelfAdjointView<Derived, UpLo> selfadjointView();
    template<unsigned int UpLo> const SelfAdjointView<Derived, UpLo> selfadjointView() const;

    static const ConstantReturnType
    Constant(int rows, int cols, const Scalar& value);
    static const ConstantReturnType
    Constant(int size, const Scalar& value);
    static const ConstantReturnType
    Constant(const Scalar& value);

    static const SequentialLinSpacedReturnType
    LinSpaced(Sequential_t, const Scalar& low, const Scalar& high, int size);
    static const RandomAccessLinSpacedReturnType
    LinSpaced(const Scalar& low, const Scalar& high, int size);

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
    Derived& setLinSpaced(const Scalar& low, const Scalar& high, int size);
    Derived& setZero();
    Derived& setOnes();
    Derived& setRandom();

    template<typename OtherDerived>
    bool isApprox(const DenseBase<OtherDerived>& other,
                  RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;
    bool isMuchSmallerThan(const RealScalar& other,
                           RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;
    template<typename OtherDerived>
    bool isMuchSmallerThan(const DenseBase<OtherDerived>& other,
                           RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;

    bool isApproxToConstant(const Scalar& value, RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;
    bool isConstant(const Scalar& value, RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;
    bool isZero(RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;
    bool isOnes(RealScalar prec = NumTraits<Scalar>::dummy_precision()) const;

    inline Derived& operator*=(const Scalar& other);
    inline Derived& operator/=(const Scalar& other);

    /** \returns the matrix or vector obtained by evaluating this expression.
      *
      * Notice that in the case of a plain matrix or vector (not an expression) this function just returns
      * a const reference, in order to avoid a useless copy.
      */
    EIGEN_STRONG_INLINE const typename ei_eval<Derived>::type eval() const
    { return typename ei_eval<Derived>::type(derived()); }

    template<typename OtherDerived>
    void swap(DenseBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other);

    /** \returns number of elements to skip to pass from one row (resp. column) to another
      * for a row-major (resp. column-major) matrix.
      * Combined with coeffRef() and the \ref flags flags, it allows a direct access to the data
      * of the underlying matrix.
      */
    inline int stride() const { return derived().stride(); }

    inline const NestByValue<Derived> nestByValue() const;
    inline const ForceAlignedAccess<Derived> forceAlignedAccess() const;
    inline ForceAlignedAccess<Derived> forceAlignedAccess();
    template<bool Enable> inline const typename ei_meta_if<Enable,ForceAlignedAccess<Derived>,Derived&>::ret forceAlignedAccessIf() const;
    template<bool Enable> inline typename ei_meta_if<Enable,ForceAlignedAccess<Derived>,Derived&>::ret forceAlignedAccessIf();

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

    inline const WithFormat<Derived> format(const IOFormat& fmt) const;

/////////// Array module ///////////

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
    select(const DenseBase<ThenDerived>& thenMatrix,
           const DenseBase<ElseDerived>& elseMatrix) const;

    template<typename ThenDerived>
    inline const Select<Derived,ThenDerived, typename ThenDerived::ConstantReturnType>
    select(const DenseBase<ThenDerived>& thenMatrix, typename ThenDerived::Scalar elseScalar) const;

    template<typename ElseDerived>
    inline const Select<Derived, typename ElseDerived::ConstantReturnType, ElseDerived >
    select(typename ElseDerived::Scalar thenScalar, const DenseBase<ElseDerived>& elseMatrix) const;

    template<int p> RealScalar lpNorm() const;

    template<int RowFactor, int ColFactor>
    const Replicate<Derived,RowFactor,ColFactor> replicate() const;
    const Replicate<Derived,Dynamic,Dynamic> replicate(int rowFacor,int colFactor) const;

    Eigen::Reverse<Derived, BothDirections> reverse();
    const Eigen::Reverse<Derived, BothDirections> reverse() const;
    void reverseInPlace();

    #ifdef EIGEN_DENSEBASE_PLUGIN
    #include EIGEN_DENSEBASE_PLUGIN
    #endif

  protected:
    /** Default constructor. Do nothing. */
    DenseBase()
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
    explicit DenseBase(int);
    DenseBase(int,int);
    template<typename OtherDerived> explicit DenseBase(const DenseBase<OtherDerived>&);
};

#endif // EIGEN_DENSEBASE_H
