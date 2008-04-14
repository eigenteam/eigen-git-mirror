// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_INVERSE_H
#define EIGEN_INVERSE_H

/** \class Inverse
  *
  * \brief Inverse of a matrix
  *
  * \param ExpressionType the type of the matrix/expression of which we are taking the inverse
  * \param CheckExistence whether or not to check the existence of the inverse while computing it
  *
  * This class represents the inverse of a matrix. It is the return
  * type of MatrixBase::inverse() and most of the time this is the only way it
  * is used.
  *
  * \sa MatrixBase::inverse(), MatrixBase::quickInverse()
  */
template<typename ExpressionType, bool CheckExistence>
struct ei_traits<Inverse<ExpressionType, CheckExistence> >
{
  typedef typename ExpressionType::Scalar Scalar;
  typedef typename ExpressionType::Eval MatrixType;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = MatrixType::Flags,
    CoeffReadCost = MatrixType::CoeffReadCost
  };
};

template<typename ExpressionType, bool CheckExistence> class Inverse : ei_no_assignment_operator,
  public MatrixBase<Inverse<ExpressionType, CheckExistence> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Inverse)
    typedef typename ei_traits<Inverse>::MatrixType MatrixType;

    Inverse(const ExpressionType& xpr)
      : m_exists(true),
        m_inverse(MatrixType::identity(xpr.rows(), xpr.cols()))
    {
      ei_assert(xpr.rows() == xpr.cols());
      _compute(xpr);
    }

    /** \returns whether or not the inverse exists.
      *
      * \note This method is only available if CheckExistence is set to true, which is the default value.
      *       For instance, when using quickInverse(), this method is not available.
      */
    bool exists() const { assert(CheckExistence); return m_exists; }

  private:

    int _rows() const { return m_inverse.rows(); }
    int _cols() const { return m_inverse.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      return m_inverse.coeff(row, col);
    }

    enum { _Size = MatrixType::RowsAtCompileTime };
    void _compute(const ExpressionType& xpr);
    void _compute_in_general_case(const ExpressionType& xpr);
    void _compute_unrolled(const ExpressionType& xpr);
    template<int Size, int Step, bool Finished = Size==Dynamic> struct _unroll_first_loop;
    template<int Size, int Step, bool Finished = Size==Dynamic> struct _unroll_second_loop;
    template<int Size, int Step, bool Finished = Size==Dynamic> struct _unroll_third_loop;

  protected:
    bool m_exists;
    MatrixType m_inverse;
};

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>
::_compute_in_general_case(const ExpressionType& xpr)
{
  MatrixType matrix(xpr);
  const RealScalar max = CheckExistence ? matrix.cwiseAbs().maxCoeff()
                                        : static_cast<RealScalar>(0);
  const int size = matrix.rows();
  for(int k = 0; k < size-1; k++)
  {
    int rowOfBiggest;
    const RealScalar max_in_this_col
      = matrix.col(k).end(size-k).cwiseAbs().maxCoeff(&rowOfBiggest);
    if(CheckExistence && ei_isMuchSmallerThan(max_in_this_col, max))
    { m_exists = false; return; }

    m_inverse.row(k).swap(m_inverse.row(k+rowOfBiggest));
    matrix.row(k).swap(matrix.row(k+rowOfBiggest));
    
    const Scalar d = matrix(k,k);
    m_inverse.block(k+1, 0, size-k-1, size)
      -= matrix.col(k).end(size-k-1) * (m_inverse.row(k) / d);
    matrix.corner(BottomRight, size-k-1, size-k)
      -= matrix.col(k).end(size-k-1) * (matrix.row(k).end(size-k) / d);
  }

  for(int k = 0; k < size-1; k++)
  {
    const Scalar d = static_cast<Scalar>(1)/matrix(k,k);
    matrix.row(k).end(size-k) *= d;
    m_inverse.row(k) *= d;
  }
  if(CheckExistence && ei_isMuchSmallerThan(matrix(size-1,size-1), max))
  { m_exists = false; return; }
  m_inverse.row(size-1) /= matrix(size-1,size-1);

  for(int k = size-1; k >= 1; k--)
  {
    m_inverse.block(0,0,k,size) -= matrix.col(k).start(k) * m_inverse.row(k);
  }
}

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>
::_compute_unrolled(const ExpressionType& xpr)
{
  MatrixType matrix(xpr);
  const RealScalar max = CheckExistence ? matrix.cwiseAbs().maxCoeff()
                                        : static_cast<RealScalar>(0);
  const int size = MatrixType::RowsAtCompileTime;
  _unroll_first_loop<size, 0>::run(*this, matrix, max);
  if(CheckExistence && !m_exists) return;
  _unroll_second_loop<size, 0>::run(*this, matrix, max);
  if(CheckExistence && !m_exists) return;
  _unroll_third_loop<size, 1>::run(*this, matrix, max);
}

template<typename ExpressionType, bool CheckExistence>
template<int Size, int Step, bool Finished>
struct Inverse<ExpressionType, CheckExistence>::_unroll_first_loop
{
  typedef Inverse<ExpressionType, CheckExistence> Inv;
  typedef typename Inv::MatrixType MatrixType;
  typedef typename Inv::RealScalar RealScalar;

  static void run(Inv& object, MatrixType& matrix, const RealScalar& max)
  {
    MatrixType& inverse = object.m_inverse;
    int rowOfBiggest;
    const RealScalar max_in_this_col
      = matrix.col(Step).template end<Size-Step>().cwiseAbs().maxCoeff(&rowOfBiggest);
    if(CheckExistence && ei_isMuchSmallerThan(max_in_this_col, max)) 
    { object.m_exists = false; return; }

    inverse.row(Step).swap(inverse.row(Step+rowOfBiggest));
    matrix.row(Step).swap(matrix.row(Step+rowOfBiggest));
  
    const Scalar d = matrix(Step,Step);
    inverse.template block<Size-Step-1, Size>(Step+1, 0)
      -= matrix.col(Step).template end<Size-Step-1>() * (inverse.row(Step) / d);
    matrix.template corner<Size-Step-1, Size-Step>(BottomRight)
      -= matrix.col(Step).template end<Size-Step-1>()
        * (matrix.row(Step).template end<Size-Step>() / d);

    _unroll_first_loop<Size, Step+1, Step >= Size-2>::run(object, matrix, max);
  }
};

template<typename ExpressionType, bool CheckExistence>
template<int Size, int Step>
struct Inverse<ExpressionType, CheckExistence>::_unroll_first_loop<Step, Size, true>
{
  typedef Inverse<ExpressionType, CheckExistence> Inv;
  typedef typename Inv::MatrixType MatrixType;
  typedef typename Inv::RealScalar RealScalar;
  static void run(Inv&, MatrixType&, const RealScalar&) {}
};

template<typename ExpressionType, bool CheckExistence>
template<int Size, int Step, bool Finished>
struct Inverse<ExpressionType, CheckExistence>::_unroll_second_loop
{
  typedef Inverse<ExpressionType, CheckExistence> Inv;
  typedef typename Inv::MatrixType MatrixType;
  typedef typename Inv::RealScalar RealScalar;

  static void run(Inv& object, MatrixType& matrix, const RealScalar& max)
  {
    MatrixType& inverse = object.m_inverse;
   
    if(Step == Size-1)
    {
      if(CheckExistence && ei_isMuchSmallerThan(matrix(Size-1,Size-1), max))
      { object.m_exists = false; return; }
      inverse.row(Size-1) /= matrix(Size-1,Size-1);
    }
    else
    {
      const Scalar d = static_cast<Scalar>(1)/matrix(Step,Step);
      matrix.row(Step).template end<Size-Step>() *= d;
      inverse.row(Step) *= d;
    }
   
    _unroll_second_loop<Size, Step+1, Step >= Size-1>::run(object, matrix, max);
  }
};

template<typename ExpressionType, bool CheckExistence>
template<int Size, int Step>
struct Inverse<ExpressionType, CheckExistence>::_unroll_second_loop <Step, Size, true>
{
  typedef Inverse<ExpressionType, CheckExistence> Inv;
  typedef typename Inv::MatrixType MatrixType;
  typedef typename Inv::RealScalar RealScalar;
  static void run(Inv&, MatrixType&, const RealScalar&) {}
};

template<typename ExpressionType, bool CheckExistence>
template<int Size, int Step, bool Finished>
struct Inverse<ExpressionType, CheckExistence>::_unroll_third_loop
{
  typedef Inverse<ExpressionType, CheckExistence> Inv;
  typedef typename Inv::MatrixType MatrixType;
  typedef typename Inv::RealScalar RealScalar;

  static void run(Inv& object, MatrixType& matrix, const RealScalar& max)
  {
    MatrixType& inverse = object.m_inverse;
    inverse.template block<Size-Step,Size>(0,0)
        -= matrix.col(Size-Step).template start<Size-Step>() * inverse.row(Size-Step);
    _unroll_third_loop<Size, Step+1, Step >= Size-1>::run(object, matrix, max);
  }
};

template<typename ExpressionType, bool CheckExistence>
template<int Size, int Step>
struct Inverse<ExpressionType, CheckExistence>::_unroll_third_loop<Step, Size, true>
{
  typedef Inverse<ExpressionType, CheckExistence> Inv;
  typedef typename Inv::MatrixType MatrixType;
  typedef typename Inv::RealScalar RealScalar;
  static void run(Inv&, MatrixType&, const RealScalar&) {}
};

template<typename ExpressionType, bool CheckExistence>
void Inverse<ExpressionType, CheckExistence>::_compute(const ExpressionType& xpr)
{
  if(_Size == 1)
  {
    const Scalar x = xpr.coeff(0,0);
    if(CheckExistence && x == static_cast<Scalar>(0))
      m_exists = false;
    else
      m_inverse.coeffRef(0,0) = static_cast<Scalar>(1) / x;
  }
  else if(_Size == 2)
  {
    const typename ei_nested<ExpressionType, 1+CheckExistence>::type matrix(xpr);
    const Scalar det = matrix.determinant();
    if(CheckExistence && ei_isMuchSmallerThan(det, matrix.cwiseAbs().maxCoeff()))
      m_exists = false;
    else
    {
      const Scalar invdet = static_cast<Scalar>(1) / det;
      m_inverse.coeffRef(0,0) = matrix.coeff(1,1) * invdet;
      m_inverse.coeffRef(1,0) = -matrix.coeff(1,0) * invdet;
      m_inverse.coeffRef(0,1) = -matrix.coeff(0,1) * invdet;
      m_inverse.coeffRef(1,1) = matrix.coeff(0,0) * invdet;
    }
  }
  else if(_Size == 3)
  {
    const typename ei_nested<ExpressionType, 2+CheckExistence>::type matrix(xpr);
    const Scalar det_minor00 = matrix.minor(0,0).determinant();
    const Scalar det_minor10 = matrix.minor(1,0).determinant();
    const Scalar det_minor20 = matrix.minor(2,0).determinant();
    const Scalar det = det_minor00 * matrix.coeff(0,0)
                      - det_minor10 * matrix.coeff(1,0)
                      + det_minor20 * matrix.coeff(2,0);
    if(CheckExistence && ei_isMuchSmallerThan(det, matrix.cwiseAbs().maxCoeff()))
      m_exists = false;
    else
    {
      const Scalar invdet = static_cast<Scalar>(1) / det;
      m_inverse.coeffRef(0, 0) = det_minor00 * invdet;
      m_inverse.coeffRef(0, 1) = -det_minor10 * invdet;
      m_inverse.coeffRef(0, 2) = det_minor20 * invdet;
      m_inverse.coeffRef(1, 0) = -matrix.minor(0,1).determinant() * invdet;
      m_inverse.coeffRef(1, 1) = matrix.minor(1,1).determinant() * invdet;
      m_inverse.coeffRef(1, 2) = -matrix.minor(2,1).determinant() * invdet;
      m_inverse.coeffRef(2, 0) = matrix.minor(0,2).determinant() * invdet;
      m_inverse.coeffRef(2, 1) = -matrix.minor(1,2).determinant() * invdet;
      m_inverse.coeffRef(2, 2) = matrix.minor(2,2).determinant() * invdet;
    }
  }
  else if(_Size == 4)
    _compute_unrolled(xpr);
  else _compute_in_general_case(xpr);
}

/** \return the matrix inverse of \c *this, if it exists.
  *
  * Example: \include MatrixBase_inverse.cpp
  * Output: \verbinclude MatrixBase_inverse.out
  *
  * \sa class Inverse
  */
template<typename Derived>
const Inverse<Derived, true>
MatrixBase<Derived>::inverse() const
{
  return Inverse<Derived, true>(derived());
}

/** \return the matrix inverse of \c *this, which is assumed to exist.
  *
  * Example: \include MatrixBase_quickInverse.cpp
  * Output: \verbinclude MatrixBase_quickInverse.out
  *
  * \sa class Inverse
  */
template<typename Derived>
const Inverse<Derived, false>
MatrixBase<Derived>::quickInverse() const
{
  return Inverse<Derived, false>(derived());
}

#endif // EIGEN_INVERSE_H
