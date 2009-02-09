// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_DIAGONALMATRIX_H
#define EIGEN_DIAGONALMATRIX_H


template<typename CoeffsVectorType, typename Derived>
class DiagonalMatrixBase : ei_no_assignment_operator,
   public MatrixBase<Derived>
{
  public:
    typedef MatrixBase<Derived> Base;
    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename Base::PacketScalar PacketScalar;
    using Base::derived;
    typedef typename ei_cleantype<CoeffsVectorType>::type _CoeffsVectorType;

  protected:

    // MSVC gets crazy if we define default parameters
    template<typename OtherDerived, bool IsVector, bool IsDiagonal> struct construct_from_expression;

    // = vector
    template<typename OtherDerived>
    struct construct_from_expression<OtherDerived,true,false>
    {
      static void run(Derived& dst, const OtherDerived& src)
      { dst.diagonal() = src; }
    };

    // = diagonal expression
    template<typename OtherDerived, bool IsVector>
    struct construct_from_expression<OtherDerived,IsVector,true>
    {
      static void run(Derived& dst, const OtherDerived& src)
      { dst.diagonal() = src.diagonal(); }
    };

    /** Default constructor without initialization */
    inline DiagonalMatrixBase() {}
    /** Constructs a diagonal matrix with given dimension */
    inline DiagonalMatrixBase(int dim) : m_coeffs(dim) {}
    /** Generic constructor from an expression */
    template<typename OtherDerived>
    inline DiagonalMatrixBase(const MatrixBase<OtherDerived>& other)
    {
      construct_from_expression<OtherDerived,OtherDerived::IsVectorAtCompileTime,(OtherDerived::Flags&Diagonal)==Diagonal>
        ::run(derived(),other.derived());
    }

  public:

    inline DiagonalMatrixBase(const _CoeffsVectorType& coeffs) : m_coeffs(coeffs)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(_CoeffsVectorType);
      ei_assert(coeffs.size() > 0);
    }

    template<typename NewType>
    inline DiagonalMatrixWrapper<NestByValue<CwiseUnaryOp<ei_scalar_cast_op<Scalar, NewType>, _CoeffsVectorType> > > cast() const
    {
      return m_coeffs.template cast<NewType>().nestByValue().asDiagonal();
    }

    /** Assignment operator.
      * The right-hand-side \a other must be either a vector representing the diagonal
      * coefficients or a diagonal matrix expression.
      */
    template<typename OtherDerived>
    inline Derived& operator=(const MatrixBase<OtherDerived>& other)
    {
      construct_from_expression<OtherDerived,OtherDerived::IsVectorAtCompileTime,(OtherDerived::Flags&Diagonal)==Diagonal>
        ::run(derived(),other);
      return derived();
    }

    inline int rows() const { return m_coeffs.size(); }
    inline int cols() const { return m_coeffs.size(); }

    inline const Scalar coeff(int row, int col) const
    {
      return row == col ? m_coeffs.coeff(row) : static_cast<Scalar>(0);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      ei_assert(row==col);
      return m_coeffs.coeffRef(row);
    }

    inline _CoeffsVectorType& diagonal() { return m_coeffs; }
    inline const _CoeffsVectorType& diagonal() const { return m_coeffs; }

  protected:
    CoeffsVectorType m_coeffs;
};

/** \class DiagonalMatrix
  * \nonstableyet
  *
  * \brief Represent a diagonal matrix with its storage
  *
  * \param _Scalar the type of coefficients
  * \param _Size the dimension of the matrix
  *
  * \sa class Matrix
  */
template<typename _Scalar,int _Size>
struct ei_traits<DiagonalMatrix<_Scalar,_Size> > : ei_traits<Matrix<_Scalar,_Size,_Size> >
{
  enum {
    Flags = (ei_traits<Matrix<_Scalar,_Size,_Size> >::Flags & HereditaryBits) | Diagonal
  };
};

template<typename _Scalar, int _Size>
class DiagonalMatrix
  : public DiagonalMatrixBase<Matrix<_Scalar,_Size,1>, DiagonalMatrix<_Scalar,_Size> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(DiagonalMatrix)
    typedef DiagonalMatrixBase<Matrix<_Scalar,_Size,1>, DiagonalMatrix<_Scalar,_Size> > DiagonalBase;

  protected:
    typedef Matrix<_Scalar,_Size,1> CoeffVectorType;
    using DiagonalBase::m_coeffs;

  public:

    /** Default constructor without initialization */
    inline DiagonalMatrix() : DiagonalBase()
    {}

    /** Constructs a diagonal matrix with given dimension  */
    inline DiagonalMatrix(int dim) : DiagonalBase(dim)
    {}

    /** 2D only */
    inline DiagonalMatrix(const Scalar& sx, const Scalar& sy)
    {
      EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DiagonalMatrix,2,2);
      m_coeffs.x() = sx;
      m_coeffs.y() = sy;
    }
    /** 3D only */
    inline DiagonalMatrix(const Scalar& sx, const Scalar& sy, const Scalar& sz)
    {
      EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DiagonalMatrix,3,3);
      m_coeffs.x() = sx;
      m_coeffs.y() = sy;
      m_coeffs.z() = sz;
    }

    /** copy constructor */
    inline DiagonalMatrix(const DiagonalMatrix& other) : DiagonalBase(other.m_coeffs)
    {}

    /** generic constructor from expression */
    template<typename OtherDerived>
    explicit inline DiagonalMatrix(const MatrixBase<OtherDerived>& other) : DiagonalBase(other)
    {}

    DiagonalMatrix& operator=(const DiagonalMatrix& other)
    {
      m_coeffs = other.m_coeffs;
      return *this;
    }

    template<typename OtherDerived>
    DiagonalMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      EIGEN_STATIC_ASSERT((OtherDerived::Flags&Diagonal)==Diagonal, THIS_METHOD_IS_ONLY_FOR_DIAGONAL_MATRIX);
      m_coeffs = other.diagonal();
      return *this;
    }
    
    inline void resize(int size)
    {
      m_coeffs.resize(size);
    }
    
    inline void resize(int rows, int cols)
    {
      ei_assert(rows==cols && "a diagonal matrix must be square");
      m_coeffs.resize(rows);
    }
    
    inline void setZero() { m_coeffs.setZero(); }
};

/** \class DiagonalMatrixWrapper
  * \nonstableyet
  *
  * \brief Expression of a diagonal matrix
  *
  * \param CoeffsVectorType the type of the vector of diagonal coefficients
  *
  * This class is an expression of a diagonal matrix with given vector of diagonal
  * coefficients. It is the return type of MatrixBase::diagonal(const OtherDerived&)
  * and most of the time this is the only way it is used.
  *
  * \sa class DiagonalMatrixBase, class DiagonalMatrix, MatrixBase::asDiagonal()
  */
template<typename CoeffsVectorType>
struct ei_traits<DiagonalMatrixWrapper<CoeffsVectorType> >
{
  typedef typename CoeffsVectorType::Scalar Scalar;
  typedef typename ei_nested<CoeffsVectorType>::type CoeffsVectorTypeNested;
  typedef typename ei_unref<CoeffsVectorTypeNested>::type _CoeffsVectorTypeNested;
  enum {
    RowsAtCompileTime = CoeffsVectorType::SizeAtCompileTime,
    ColsAtCompileTime = CoeffsVectorType::SizeAtCompileTime,
    MaxRowsAtCompileTime = CoeffsVectorType::MaxSizeAtCompileTime,
    MaxColsAtCompileTime = CoeffsVectorType::MaxSizeAtCompileTime,
    Flags = (_CoeffsVectorTypeNested::Flags & HereditaryBits) | Diagonal,
    CoeffReadCost = _CoeffsVectorTypeNested::CoeffReadCost
  };
};
template<typename CoeffsVectorType>
class DiagonalMatrixWrapper
  : public DiagonalMatrixBase<typename CoeffsVectorType::Nested, DiagonalMatrixWrapper<CoeffsVectorType> >
{
    typedef typename CoeffsVectorType::Nested CoeffsVectorTypeNested;
    typedef DiagonalMatrixBase<CoeffsVectorTypeNested, DiagonalMatrixWrapper<CoeffsVectorType> > DiagonalBase;
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(DiagonalMatrixWrapper)
    inline DiagonalMatrixWrapper(const CoeffsVectorType& coeffs) : DiagonalBase(coeffs)
    {}
};

/** \nonstableyet
  * \returns an expression of a diagonal matrix with *this as vector of diagonal coefficients
  *
  * \only_for_vectors
  *
  * \addexample AsDiagonalExample \label How to build a diagonal matrix from a vector
  *
  * Example: \include MatrixBase_asDiagonal.cpp
  * Output: \verbinclude MatrixBase_asDiagonal.out
  *
  * \sa class DiagonalMatrix, isDiagonal()
  **/
template<typename Derived>
inline const DiagonalMatrixWrapper<Derived>
MatrixBase<Derived>::asDiagonal() const
{
  return derived();
}

/** \nonstableyet
  * \returns true if *this is approximately equal to a diagonal matrix,
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isDiagonal.cpp
  * Output: \verbinclude MatrixBase_isDiagonal.out
  *
  * \sa asDiagonal()
  */
template<typename Derived>
bool MatrixBase<Derived>::isDiagonal
(RealScalar prec) const
{
  if(cols() != rows()) return false;
  RealScalar maxAbsOnDiagonal = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); ++j)
  {
    RealScalar absOnDiagonal = ei_abs(coeff(j,j));
    if(absOnDiagonal > maxAbsOnDiagonal) maxAbsOnDiagonal = absOnDiagonal;
  }
  for(int j = 0; j < cols(); ++j)
    for(int i = 0; i < j; ++i)
    {
      if(!ei_isMuchSmallerThan(coeff(i, j), maxAbsOnDiagonal, prec)) return false;
      if(!ei_isMuchSmallerThan(coeff(j, i), maxAbsOnDiagonal, prec)) return false;
    }
  return true;
}

#endif // EIGEN_DIAGONALMATRIX_H
