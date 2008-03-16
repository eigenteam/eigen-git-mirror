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

#ifndef EIGEN_MATRIXBASE_H
#define EIGEN_MATRIXBASE_H

/** \class MatrixBase
  *
  * \brief Base class for all matrices, vectors, and expressions
  *
  * This class is the base that is inherited by all matrix, vector, and expression
  * types. Most of the Eigen API is contained in this class.
  *
  * \param Derived is the derived type, e.g. a matrix type, or an expression, etc.
  *
  * When writing a function taking Eigen objects as argument, if you want your function
  * to take as argument any matrix, vector, or expression, just let it take a
  * MatrixBase argument. As an example, here is a function printFirstRow which, given
  * a matrix, vector, or expression \a x, prints the first row of \a x.
  *
  * \code
    template<typename Derived>
    void printFirstRow(const Eigen::MatrixBase<Derived>& x)
    {
      cout << x.row(0) << endl;
    }
  * \endcode
  *
  * \nosubgrouping
  */
template<typename Derived> class MatrixBase
{
    struct CommaInitializer;

  public:

    /// \name Compile-time traits
    //@{
    typedef typename ei_traits<Derived>::Scalar Scalar;

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

      SizeAtCompileTime
        = ei_traits<Derived>::RowsAtCompileTime == Dynamic
        || ei_traits<Derived>::ColsAtCompileTime == Dynamic
        ? Dynamic
        : ei_traits<Derived>::RowsAtCompileTime * ei_traits<Derived>::ColsAtCompileTime,
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

      MaxSizeAtCompileTime
        = ei_traits<Derived>::MaxRowsAtCompileTime == Dynamic
        || ei_traits<Derived>::MaxColsAtCompileTime == Dynamic
        ? Dynamic
        : ei_traits<Derived>::MaxRowsAtCompileTime * ei_traits<Derived>::MaxColsAtCompileTime,
        /**< This value is equal to the maximum possible number of coefficients that this expression
          * might have. If this expression might have an arbitrarily high number of coefficients,
          * this value is set to \a Dynamic.
          *
          * This value is useful to know when evaluating an expression, in order to determine
          * whether it is possible to avoid doing a dynamic memory allocation.
          *
          * \sa SizeAtCompileTime, MaxRowsAtCompileTime, MaxColsAtCompileTime
          */

      IsVectorAtCompileTime
        = ei_traits<Derived>::RowsAtCompileTime == 1 || ei_traits<Derived>::ColsAtCompileTime == 1
        /**< This is set to true if either the number of rows or the number of
          * columns is known at compile-time to be equal to 1. Indeed, in that case,
          * we are dealing with a column-vector (if there is only one column) or with
          * a row-vector (if there is only one row). */
    };

    /** This is the "real scalar" type; if the \a Scalar type is already real numbers
      * (e.g. int, float or double) then \a RealScalar is just the same as \a Scalar. If
      * \a Scalar is \a std::complex<T> then RealScalar is \a T.
      *
      * In fact, \a RealScalar is defined as follows:
      * \code typedef typename NumTraits<Scalar>::Real RealScalar; \endcode
      *
      * \sa class NumTraits
      */
    typedef typename NumTraits<Scalar>::Real RealScalar;
    //@}

    /// \name Run-time traits
    //@{
    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    int rows() const { return derived()._rows(); }
    /** \returns the number of columns. \sa row(), ColsAtCompileTime*/
    int cols() const { return derived()._cols(); }
    /** \returns the number of coefficients, which is \a rows()*cols().
      * \sa rows(), cols(), SizeAtCompileTime. */
    int size() const { return rows() * cols(); }
    /** \returns true if either the number of rows or the number of columns is equal to 1.
      * In other words, this function returns
      * \code rows()==1 || cols()==1 \endcode
      * \sa rows(), cols(), IsVectorAtCompileTime. */
    bool isVector() const { return rows()==1 || cols()==1; }
    //@}

    /// \name Copying and initialization
    //@{

    /** Copies \a other into *this. \returns a reference to *this. */
    template<typename OtherDerived>
    Derived& operator=(const MatrixBase<OtherDerived>& other);

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    Derived& operator=(const MatrixBase& other)
    {
      return this->operator=<Derived>(other);
    }

    CommaInitializer operator<< (const Scalar& s);

    template<typename OtherDerived>
    CommaInitializer operator<< (const MatrixBase<OtherDerived>& other);
    //@}

    /// \name Coefficient accessors
    //@{
    Scalar coeff(int row, int col) const;
    Scalar operator()(int row, int col) const;

    Scalar& coeffRef(int row, int col);
    Scalar& operator()(int row, int col);

    Scalar coeff(int index) const;
    Scalar operator[](int index) const;

    Scalar& coeffRef(int index);
    Scalar& operator[](int index);

    Scalar x() const;
    Scalar y() const;
    Scalar z() const;
    Scalar w() const;
    Scalar& x();
    Scalar& y();
    Scalar& z();
    Scalar& w();
    //@}

    /** \name Linear structure
      * sum, scalar multiple, ...
      */
    //@{
    const CwiseUnaryOp<ei_scalar_opposite_op,Derived> operator-() const;

    template<typename OtherDerived>
    const CwiseBinaryOp<ei_scalar_sum_op, Derived, OtherDerived>
    operator+(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const CwiseBinaryOp<ei_scalar_difference_op, Derived, OtherDerived>
    operator-(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    Derived& operator+=(const MatrixBase<OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const MatrixBase<OtherDerived>& other);

    Derived& operator*=(const Scalar& other);
    Derived& operator/=(const Scalar& other);

    const CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived> operator*(const Scalar& scalar) const;
    const CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived> operator/(const Scalar& scalar) const;

    friend const CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived>
    operator*(const Scalar& scalar, const MatrixBase& matrix)
    { return matrix*scalar; }
    //@}

    /** \name Matrix product
      */
    //@{
    template<typename OtherDerived>
    const Product<Derived, OtherDerived>
    lazyProduct(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    const Eval<Product<Derived, OtherDerived> >
    operator*(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    Derived& operator*=(const MatrixBase<OtherDerived>& other);
    //@}

    /** \name Dot product and related notions
      * including vector norm, adjoint, transpose ...
      */
    //@{
    template<typename OtherDerived>
    Scalar dot(const MatrixBase<OtherDerived>& other) const;
    RealScalar norm2() const;
    RealScalar norm()  const;
    const CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, Derived> normalized() const;

    Transpose<Derived> transpose();
    const Transpose<Derived> transpose() const;
    const Transpose<CwiseUnaryOp<ei_scalar_conjugate_op, Derived> > adjoint() const;
    //@}

    /// \name Sub-matrices
    //@{
    Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime> row(int i);
    const Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime> row(int i) const;

    Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1> col(int i);
    const Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1> col(int i) const;

    Minor<Derived> minor(int row, int col);
    const Minor<Derived> minor(int row, int col) const;

    Block<Derived> block(int startRow, int startCol, int blockRows, int blockCols);
    const Block<Derived>
    block(int startRow, int startCol, int blockRows, int blockCols) const;

    Block<Derived> block(int start, int size);
    const Block<Derived> block(int start, int size) const;

    Block<Derived> start(int size);
    const Block<Derived> start(int size) const;

    Block<Derived> end(int size);
    const Block<Derived> end(int size) const;

    Block<Derived> corner(CornerType type, int cRows, int cCols);
    const Block<Derived> corner(CornerType type, int cRows, int cCols) const;

    template<int BlockRows, int BlockCols>
    Block<Derived, BlockRows, BlockCols> block(int startRow, int startCol);
    template<int BlockRows, int BlockCols>
    const Block<Derived, BlockRows, BlockCols> block(int startRow, int startCol) const;

    DiagonalCoeffs<Derived> diagonal();
    const DiagonalCoeffs<Derived> diagonal() const;
    //@}

    /// \name Generating special matrices
    //@{
    static const Eval<Random<Derived> > random(int rows, int cols);
    static const Eval<Random<Derived> > random(int size);
    static const Eval<Random<Derived> > random();
    static const Zero<Derived> zero(int rows, int cols);
    static const Zero<Derived> zero(int size);
    static const Zero<Derived> zero();
    static const Ones<Derived> ones(int rows, int cols);
    static const Ones<Derived> ones(int size);
    static const Ones<Derived> ones();
    static const Identity<Derived> identity();
    static const Identity<Derived> identity(int rows, int cols);

    const DiagonalMatrix<Derived> asDiagonal() const;

    Derived& setZero();
    Derived& setOnes();
    Derived& setRandom();
    Derived& setIdentity();
    //@}

    /// \name Comparison and diagnostic
    //@{
    template<typename OtherDerived>
    bool isApprox(const OtherDerived& other,
                  RealScalar prec = precision<Scalar>()) const;
    bool isMuchSmallerThan(const RealScalar& other,
                           RealScalar prec = precision<Scalar>()) const;
    template<typename OtherDerived>
    bool isMuchSmallerThan(const MatrixBase<OtherDerived>& other,
                           RealScalar prec = precision<Scalar>()) const;

    bool isZero(RealScalar prec = precision<Scalar>()) const;
    bool isOnes(RealScalar prec = precision<Scalar>()) const;
    bool isIdentity(RealScalar prec = precision<Scalar>()) const;
    bool isDiagonal(RealScalar prec = precision<Scalar>()) const;

    template<typename OtherDerived>
    bool isOrtho(const MatrixBase<OtherDerived>& other,
                 RealScalar prec = precision<Scalar>()) const;
    bool isOrtho(RealScalar prec = precision<Scalar>()) const;

    /** puts in *row and *col the location of the coefficient of *this
      * which has the biggest absolute value.
      */
    void findBiggestCoeff(int *row, int *col) const
    { (*this).cwiseAbs().maxCoeff(row, col); }
    //@}

    /// \name Special functions
    //@{
    template<typename NewType>
    const CwiseUnaryOp<ei_scalar_cast_op<NewType>, Derived> cast() const;

    const Eval<Derived> eval() const EIGEN_ALWAYS_INLINE;
    const EvalOMP<Derived> evalOMP() const EIGEN_ALWAYS_INLINE;

    /** swaps *this with the expression \a other.
      *
      * \note \a other is only marked const because I couln't find another way
      * to get g++ 4.2 to accept that template parameter resolution. It gets const_cast'd
      * of course. TODO: get rid of const here.
      */
    template<typename OtherDerived>
    void swap(const MatrixBase<OtherDerived>& other);
    //@}

    /// \name Coefficient-wise operations
    //@{
    const CwiseUnaryOp<ei_scalar_conjugate_op, Derived> conjugate() const;

    template<typename OtherDerived>
    const CwiseBinaryOp<ei_scalar_product_op, Derived, OtherDerived>
    cwiseProduct(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const CwiseBinaryOp<ei_scalar_quotient_op, Derived, OtherDerived>
    cwiseQuotient(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const CwiseBinaryOp<ei_scalar_min_op, Derived, OtherDerived>
    cwiseMin(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    const CwiseBinaryOp<ei_scalar_max_op, Derived, OtherDerived>
    cwiseMax(const MatrixBase<OtherDerived> &other) const;

    const CwiseUnaryOp<ei_scalar_abs_op, Derived> cwiseAbs() const;
    const CwiseUnaryOp<ei_scalar_abs2_op, Derived> cwiseAbs2() const;
    const CwiseUnaryOp<ei_scalar_sqrt_op, Derived> cwiseSqrt() const;
    const CwiseUnaryOp<ei_scalar_exp_op, Derived> cwiseExp() const;
    const CwiseUnaryOp<ei_scalar_log_op, Derived> cwiseLog() const;
    const CwiseUnaryOp<ei_scalar_cos_op, Derived> cwiseCos() const;
    const CwiseUnaryOp<ei_scalar_sin_op, Derived> cwiseSin() const;
    const CwiseUnaryOp<ei_scalar_pow_op<typename ei_traits<Derived>::Scalar>, Derived>
    cwisePow(const Scalar& exponent) const;

    template<typename CustomUnaryOp>
    const CwiseUnaryOp<CustomUnaryOp, Derived> cwise(const CustomUnaryOp& func = CustomUnaryOp()) const;

    template<typename CustomBinaryOp, typename OtherDerived>
    const CwiseBinaryOp<CustomBinaryOp, Derived, OtherDerived>
    cwise(const MatrixBase<OtherDerived> &other, const CustomBinaryOp& func = CustomBinaryOp()) const;
    //@}

    /// \name Redux and visitor
    //@{
    Scalar sum() const;
    Scalar trace() const;

    typename ei_traits<Derived>::Scalar minCoeff() const;
    typename ei_traits<Derived>::Scalar maxCoeff() const;

    typename ei_traits<Derived>::Scalar minCoeff(int* row, int* col = 0) const;
    typename ei_traits<Derived>::Scalar maxCoeff(int* row, int* col = 0) const;

    template<typename BinaryOp>
    const PartialRedux<Vertical, BinaryOp, Derived>
    verticalRedux(const BinaryOp& func) const;

    template<typename BinaryOp>
    const PartialRedux<Horizontal, BinaryOp, Derived>
    horizontalRedux(const BinaryOp& func) const;

    template<typename BinaryOp>
    typename ei_result_of<BinaryOp(typename ei_traits<Derived>::Scalar)>::type
    redux(const BinaryOp& func) const;

    template<typename Visitor>
    void visit(Visitor& func) const;
    //@}

    /// \name Casting to the derived type
    //@{
    const Derived& derived() const { return *static_cast<const Derived*>(this); }
    Derived& derived() { return *static_cast<Derived*>(this); }
    Derived& const_cast_derived() const
    { return *static_cast<Derived*>(const_cast<MatrixBase*>(this)); }
    //@}

};

#endif // EIGEN_MATRIXBASE_H
