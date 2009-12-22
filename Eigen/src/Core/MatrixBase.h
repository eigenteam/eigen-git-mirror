// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
  * \brief Base class for all dense matrices, vectors, and expressions
  *
  * This class is the base that is inherited by all matrix, vector, and expression
  * types. Most of the Eigen API is contained in this class. Other important classes for
  * the Eigen API are Matrix, Cwise, and VectorwiseOp.
  *
  * Note that some methods are defined in the \ref Array_Module array module.
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
  */
template<typename Derived> class MatrixBase
  : public DenseBase<Derived>
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** The base class for a given storage type. */
    typedef MatrixBase StorageBaseType;
    /** Construct the base class type for the derived class OtherDerived */
    template <typename OtherDerived> struct MakeBase { typedef MatrixBase<OtherDerived> Type; };

//     using DenseBase<Derived>::operator*;

    class InnerIterator;

    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename ei_packet_traits<Scalar>::type PacketScalar;

    typedef DenseBase<Derived> Base;

    using Base::RowsAtCompileTime;
    using Base::ColsAtCompileTime;
    using Base::SizeAtCompileTime;
    using Base::MaxRowsAtCompileTime;
    using Base::MaxColsAtCompileTime;
    using Base::MaxSizeAtCompileTime;
    using Base::IsVectorAtCompileTime;
    using Base::Flags;
    using Base::CoeffReadCost;
    using Base::_HasDirectAccess;

    using Base::derived;
    using Base::const_cast_derived;
    using Base::rows;
    using Base::cols;
    using Base::size;
    using Base::coeff;
    using Base::coeffRef;
    using Base::lazyAssign;
    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;

    typedef typename Base::CoeffReturnType CoeffReturnType;
#endif // not EIGEN_PARSED_BY_DOXYGEN



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

    /** \returns the size of the main diagonal, which is min(rows(),cols()).
      * \sa rows(), cols(), SizeAtCompileTime. */
    inline int diagonalSize() const { return std::min(rows(),cols()); }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal the plain matrix type corresponding to this expression. Note that is not necessarily
      * exactly the return type of eval(): in the case of plain matrices, the return type of eval() is a const
      * reference to a matrix, not a matrix! It is however guaranteed that the return type of eval() is either
      * PlainMatrixType or const PlainMatrixType&.
      */
//     typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType;
    typedef Matrix<typename ei_traits<Derived>::Scalar,
                ei_traits<Derived>::RowsAtCompileTime,
                ei_traits<Derived>::ColsAtCompileTime,
                AutoAlign | (ei_traits<Derived>::Flags&RowMajorBit ? RowMajor : ColMajor),
                ei_traits<Derived>::MaxRowsAtCompileTime,
                ei_traits<Derived>::MaxColsAtCompileTime
          > PlainMatrixType;
    /** \internal the column-major plain matrix type corresponding to this expression. Note that is not necessarily
      * exactly the return type of eval(): in the case of plain matrices, the return type of eval() is a const
      * reference to a matrix, not a matrix!
      * The only difference from PlainMatrixType is that PlainMatrixType_ColMajor is guaranteed to be column-major.
      */
//     typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType_ColMajor;


    /** \internal Represents a matrix with all coefficients equal to one another*/
    typedef CwiseNullaryOp<ei_scalar_constant_op<Scalar>,Derived> ConstantReturnType;
    /** \internal the return type of MatrixBase::adjoint() */
    typedef typename ei_meta_if<NumTraits<Scalar>::IsComplex,
                        CwiseUnaryOp<ei_scalar_conjugate_op<Scalar>, Eigen::Transpose<Derived> >,
                        Transpose<Derived>
                     >::ret AdjointReturnType;
    /** \internal the return type of MatrixBase::eigenvalues() */
    typedef Matrix<typename NumTraits<typename ei_traits<Derived>::Scalar>::Real, ei_traits<Derived>::ColsAtCompileTime, 1> EigenvaluesReturnType;
    /** \internal expression tyepe of a column */
    typedef Block<Derived, ei_traits<Derived>::RowsAtCompileTime, 1> ColXpr;
    /** \internal expression tyepe of a column */
    typedef Block<Derived, 1, ei_traits<Derived>::ColsAtCompileTime> RowXpr;
    /** \internal the return type of identity */
    typedef CwiseNullaryOp<ei_scalar_identity_op<Scalar>,Derived> IdentityReturnType;
    /** \internal the return type of unit vectors */
    typedef Block<CwiseNullaryOp<ei_scalar_identity_op<Scalar>, SquareMatrixType>,
                  ei_traits<Derived>::RowsAtCompileTime,
                  ei_traits<Derived>::ColsAtCompileTime> BasisReturnType;
#endif // not EIGEN_PARSED_BY_DOXYGEN

#define EIGEN_CURRENT_STORAGE_BASE_CLASS Eigen::MatrixBase
#   include "../plugins/CommonCwiseUnaryOps.h"
#   include "../plugins/CommonCwiseBinaryOps.h"
#   include "../plugins/MatrixCwiseUnaryOps.h"
#   include "../plugins/MatrixCwiseBinaryOps.h"
#   ifdef EIGEN_MATRIXBASE_PLUGIN
#     include EIGEN_MATRIXBASE_PLUGIN
#   endif
#undef EIGEN_CURRENT_STORAGE_BASE_CLASS

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    Derived& operator=(const MatrixBase& other);

#ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename ProductDerived, typename Lhs, typename Rhs>
    Derived& lazyAssign(const ProductBase<ProductDerived, Lhs,Rhs>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    const CoeffReturnType x() const;
    const CoeffReturnType y() const;
    const CoeffReturnType z() const;
    const CoeffReturnType w() const;
    Scalar& x();
    Scalar& y();
    Scalar& z();
    Scalar& w();

    template<typename OtherDerived>
    Derived& operator+=(const MatrixBase<OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const MatrixBase<OtherDerived>& other);

    template<typename OtherDerived>
    const typename ProductReturnType<Derived,OtherDerived>::Type
    operator*(const MatrixBase<OtherDerived> &other) const;

    template<typename OtherDerived>
    Derived& operator*=(const AnyMatrixBase<OtherDerived>& other);

    template<typename OtherDerived>
    void applyOnTheLeft(const AnyMatrixBase<OtherDerived>& other);

    template<typename OtherDerived>
    void applyOnTheRight(const AnyMatrixBase<OtherDerived>& other);

    template<typename DiagonalDerived>
    const DiagonalProduct<Derived, DiagonalDerived, OnTheRight>
    operator*(const DiagonalBase<DiagonalDerived> &diagonal) const;

    template<typename OtherDerived>
    Scalar dot(const MatrixBase<OtherDerived>& other) const;
    RealScalar squaredNorm() const;
    RealScalar norm() const;
    RealScalar stableNorm() const;
    RealScalar blueNorm() const;
    RealScalar hypotNorm() const;
    const PlainMatrixType normalized() const;
    void normalize();

    const AdjointReturnType adjoint() const;
    void adjointInPlace();

    Minor<Derived> minor(int row, int col);
    const Minor<Derived> minor(int row, int col) const;

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

    static const IdentityReturnType Identity();
    static const IdentityReturnType Identity(int rows, int cols);
    static const BasisReturnType Unit(int size, int i);
    static const BasisReturnType Unit(int i);
    static const BasisReturnType UnitX();
    static const BasisReturnType UnitY();
    static const BasisReturnType UnitZ();
    static const BasisReturnType UnitW();

    const DiagonalWrapper<Derived> asDiagonal() const;

    Derived& setIdentity();

    bool isIdentity(RealScalar prec = dummy_precision<Scalar>()) const;
    bool isDiagonal(RealScalar prec = dummy_precision<Scalar>()) const;

    bool isUpperTriangular(RealScalar prec = dummy_precision<Scalar>()) const;
    bool isLowerTriangular(RealScalar prec = dummy_precision<Scalar>()) const;

    template<typename OtherDerived>
    bool isOrthogonal(const MatrixBase<OtherDerived>& other,
                      RealScalar prec = dummy_precision<Scalar>()) const;
    bool isUnitary(RealScalar prec = dummy_precision<Scalar>()) const;

    /** \returns true if each coefficients of \c *this and \a other are all exactly equal.
      * \warning When using floating point scalar values you probably should rather use a
      *          fuzzy comparison such as isApprox()
      * \sa isApprox(), operator!= */
    template<typename OtherDerived>
    inline bool operator==(const MatrixBase<OtherDerived>& other) const
    { return cwiseEqual(other).all(); }

    /** \returns true if at least one pair of coefficients of \c *this and \a other are not exactly equal to each other.
      * \warning When using floating point scalar values you probably should rather use a
      *          fuzzy comparison such as isApprox()
      * \sa isApprox(), operator== */
    template<typename OtherDerived>
    inline bool operator!=(const MatrixBase<OtherDerived>& other) const
    { return cwiseNotEqual(other).all(); }


    /** \returns the matrix or vector obtained by evaluating this expression.
      *
      * Notice that in the case of a plain matrix or vector (not an expression) this function just returns
      * a const reference, in order to avoid a useless copy.
      */
    EIGEN_STRONG_INLINE const typename ei_eval<Derived>::type eval() const
    { return typename ei_eval<Derived>::type(derived()); }

    template<typename OtherDerived>
    void swap(MatrixBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other);

    NoAlias<Derived,Eigen::MatrixBase > noalias();

    inline const NestByValue<Derived> nestByValue() const;
    inline const ForceAlignedAccess<Derived> forceAlignedAccess() const;
    inline ForceAlignedAccess<Derived> forceAlignedAccess();
    template<bool Enable> inline const typename ei_meta_if<Enable,ForceAlignedAccess<Derived>,Derived&>::ret forceAlignedAccessIf() const;
    template<bool Enable> inline typename ei_meta_if<Enable,ForceAlignedAccess<Derived>,Derived&>::ret forceAlignedAccessIf();

    Scalar mean() const;
    Scalar trace() const;

/////////// Array module ///////////

    const VectorwiseOp<Derived,Horizontal> rowwise() const;
    VectorwiseOp<Derived,Horizontal> rowwise();
    const VectorwiseOp<Derived,Vertical> colwise() const;
    VectorwiseOp<Derived,Vertical> colwise();

    template<int p> RealScalar lpNorm() const;

    ArrayWrapper<Derived> array() { return derived(); }
    const ArrayWrapper<Derived> array() const { return derived(); }

/////////// LU module ///////////

    const FullPivLU<PlainMatrixType> fullPivLu() const;
    const PartialPivLU<PlainMatrixType> partialPivLu() const;
    const PartialPivLU<PlainMatrixType> lu() const;
    const ei_inverse_impl<Derived> inverse() const;
    template<typename ResultType>
    void computeInverseAndDetWithCheck(
      ResultType& inverse,
      typename ResultType::Scalar& determinant,
      bool& invertible,
      const RealScalar& absDeterminantThreshold = dummy_precision<Scalar>()
    ) const;
    template<typename ResultType>
    void computeInverseWithCheck(
      ResultType& inverse,
      bool& invertible,
      const RealScalar& absDeterminantThreshold = dummy_precision<Scalar>()
    ) const;
    Scalar determinant() const;

/////////// Cholesky module ///////////

    const LLT<PlainMatrixType>  llt() const;
    const LDLT<PlainMatrixType> ldlt() const;

/////////// QR module ///////////

    const HouseholderQR<PlainMatrixType> householderQr() const;
    const ColPivHouseholderQR<PlainMatrixType> colPivHouseholderQr() const;
    const FullPivHouseholderQR<PlainMatrixType> fullPivHouseholderQr() const;

    EigenvaluesReturnType eigenvalues() const;
    RealScalar operatorNorm() const;

/////////// SVD module ///////////

    SVD<PlainMatrixType> svd() const;

/////////// Geometry module ///////////

    template<typename OtherDerived>
    PlainMatrixType cross(const MatrixBase<OtherDerived>& other) const;
    template<typename OtherDerived>
    PlainMatrixType cross3(const MatrixBase<OtherDerived>& other) const;
    PlainMatrixType unitOrthogonal(void) const;
    Matrix<Scalar,3,1> eulerAngles(int a0, int a1, int a2) const;
    const ScalarMultipleReturnType operator*(const UniformScaling<Scalar>& s) const;
    enum {
      SizeMinusOne = SizeAtCompileTime==Dynamic ? Dynamic : SizeAtCompileTime-1
    };
    typedef Block<Derived,
                  ei_traits<Derived>::ColsAtCompileTime==1 ? SizeMinusOne : 1,
                  ei_traits<Derived>::ColsAtCompileTime==1 ? 1 : SizeMinusOne> StartMinusOne;
    typedef CwiseUnaryOp<ei_scalar_quotient1_op<typename ei_traits<Derived>::Scalar>,
                StartMinusOne > HNormalizedReturnType;

    const HNormalizedReturnType hnormalized() const;
    typedef Homogeneous<Derived,MatrixBase<Derived>::ColsAtCompileTime==1?Vertical:Horizontal> HomogeneousReturnType;
    const HomogeneousReturnType homogeneous() const;

////////// Householder module ///////////

    void makeHouseholderInPlace(Scalar& tau, RealScalar& beta);
    template<typename EssentialPart>
    void makeHouseholder(EssentialPart& essential,
                         Scalar& tau, RealScalar& beta) const;
    template<typename EssentialPart>
    void applyHouseholderOnTheLeft(const EssentialPart& essential,
                                   const Scalar& tau,
                                   Scalar* workspace);
    template<typename EssentialPart>
    void applyHouseholderOnTheRight(const EssentialPart& essential,
                                    const Scalar& tau,
                                    Scalar* workspace);

///////// Jacobi module /////////

    template<typename OtherScalar>
    void applyOnTheLeft(int p, int q, const PlanarRotation<OtherScalar>& j);
    template<typename OtherScalar>
    void applyOnTheRight(int p, int q, const PlanarRotation<OtherScalar>& j);

#ifdef EIGEN2_SUPPORT
    template<typename ProductDerived, typename Lhs, typename Rhs>
    Derived& operator+=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                      EvalBeforeAssigningBit>& other);

    template<typename ProductDerived, typename Lhs, typename Rhs>
    Derived& operator-=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                      EvalBeforeAssigningBit>& other);

    /** \deprecated because .lazy() is deprecated
      * Overloaded for cache friendly product evaluation */
    template<typename OtherDerived>
    Derived& lazyAssign(const Flagged<OtherDerived, 0, EvalBeforeAssigningBit>& other)
    { return lazyAssign(other._expression()); }

    template<unsigned int Added>
    const Flagged<Derived, Added, 0> marked() const;
    const Flagged<Derived, 0, EvalBeforeAssigningBit> lazy() const;

    inline const Cwise<Derived> cwise() const;
    inline Cwise<Derived> cwise();

    template<typename OtherDerived>
    typename ei_plain_matrix_type_column_major<OtherDerived>::type
    solveTriangular(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    void solveTriangularInPlace(const MatrixBase<OtherDerived>& other) const;
#endif

  protected:
    /** Default constructor. Do nothing. */
    MatrixBase()
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
    explicit MatrixBase(int);
    MatrixBase(int,int);
    template<typename OtherDerived> explicit MatrixBase(const MatrixBase<OtherDerived>&);
};

#endif // EIGEN_MATRIXBASE_H
