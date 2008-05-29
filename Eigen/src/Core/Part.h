// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_PART_H
#define EIGEN_PART_H

/** \class Part
  *
  * \brief Pseudo-expression allowing to write to a special part of a matrix
  *
  * This lvalue-only pseudo-expression allows to perform special operations
  * on a matrix, such as writing only to the upper (above diagonal) part.
  *
  * It is the return type of MatrixBase::part() and most of the time this is
  * the only way that it is used.
  *
  * \sa class Extract, MatrixBase::part()
  */
template<typename MatrixType, unsigned int Mode>
class Part
{
  public:
    Part(MatrixType& matrix);
    /** \sa operator=(), MatrixBase::lazyAssign() */
    template<typename Other> void lazyAssign(const Other& other);
    /** \sa MatrixBase::operator=() */
    template<typename Other> void operator=(const Other& other);
    /** \sa MatrixBase::operator+=() */
    template<typename Other> void operator+=(const Other& other);
    /** \sa MatrixBase::operator-=() */
    template<typename Other> void operator-=(const Other& other);
    /** \sa MatrixBase::operator*=() */
    void operator*=(const typename ei_traits<MatrixType>::Scalar& other);
    /** \sa MatrixBase::operator/=() */
    void operator/=(const typename ei_traits<MatrixType>::Scalar& other);
    /** \sa MatrixBase::setConstant() */
    void setConstant(const typename ei_traits<MatrixType>::Scalar& value);
    /** \sa MatrixBase::setZero() */
    void setZero();
    /** \sa MatrixBase::setOnes() */
    void setOnes();
    /** \sa MatrixBase::setRandom() */
    void setRandom();
    /** \sa MatrixBase::setIdentity() */
    void setIdentity();

  private:
    MatrixType& m_matrix;
};

template<typename MatrixType, unsigned int Mode>
inline Part<MatrixType, Mode>::Part(MatrixType& matrix)
 : m_matrix(matrix)
{
  ei_assert(ei_are_flags_consistent<Mode>::ret);
}

template<typename MatrixType, unsigned int Mode>
template<typename Other>
inline void Part<MatrixType, Mode>::operator=(const Other& other)
{
  if(Other::Flags & EvalBeforeAssigningBit)
  {
    typename ei_eval<Other>::type other_evaluated(other.rows(), other.cols());
    other_evaluated.template part<Mode>().lazyAssign(other);
    lazyAssign(other_evaluated);
  }
  else
    lazyAssign(other.derived());
}

template<typename Derived1, typename Derived2, unsigned int Mode, int UnrollCount>
struct ei_part_assignment_unroller
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_part_assignment_unroller<Derived1, Derived2, Mode, UnrollCount-1>::run(dst, src);

    if(Mode == SelfAdjoint)
    {
      if(row == col)
	dst.coeffRef(row, col) = ei_real(src.coeff(row, col));
      else if(row < col)
	dst.coeffRef(col, row) = ei_conj(dst.coeffRef(row, col) = src.coeff(row, col));
    }
    else
    {
      if((Mode == Upper && row <= col)
      || (Mode == Lower && row >= col)
      || (Mode == StrictlyUpper && row < col)
      || (Mode == StrictlyLower && row > col))
	dst.coeffRef(row, col) = src.coeff(row, col);
    }
  }
};

template<typename Derived1, typename Derived2, unsigned int Mode>
struct ei_part_assignment_unroller<Derived1, Derived2, Mode, 1>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    if(!(Mode & ZeroDiagBit))
      dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, unsigned int Mode>
struct ei_part_assignment_unroller<Derived1, Derived2, Mode, 0>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, unsigned int Mode>
struct ei_part_assignment_unroller<Derived1, Derived2, Mode, Dynamic>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};


template<typename MatrixType, unsigned int Mode>
template<typename Other>
void Part<MatrixType, Mode>::lazyAssign(const Other& other)
{
  const bool unroll = MatrixType::SizeAtCompileTime * Other::CoeffReadCost / 2 <= EIGEN_UNROLLING_LIMIT;
  ei_assert(m_matrix.rows() == other.rows() && m_matrix.cols() == other.cols());
  if(unroll)
  {
    ei_part_assignment_unroller
      <MatrixType, Other, Mode,
      unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic
      >::run(m_matrix, other.derived());
  }
  else
  {
    switch(Mode)
    {
      case Upper:
        for(int j = 0; j < m_matrix.cols(); j++)
          for(int i = 0; i <= j; i++)
            m_matrix.coeffRef(i, j) = other.coeff(i, j);
        break;
      case Lower:
        for(int j = 0; j < m_matrix.cols(); j++)
          for(int i = j; i < m_matrix.rows(); i++)
            m_matrix.coeffRef(i, j) = other.coeff(i, j);
        break;
      case StrictlyUpper:
        for(int j = 0; j < m_matrix.cols(); j++)
          for(int i = 0; i < j; i++)
            m_matrix.coeffRef(i, j) = other.coeff(i, j);
        break;
      case StrictlyLower:
        for(int j = 0; j < m_matrix.cols(); j++)
          for(int i = j+1; i < m_matrix.rows(); i++)
            m_matrix.coeffRef(i, j) = other.coeff(i, j);
        break;
      case SelfAdjoint:
        for(int j = 0; j < m_matrix.cols(); j++)
        {
          for(int i = 0; i < j; i++)
            m_matrix.coeffRef(j, i) = ei_conj(m_matrix.coeffRef(i, j) = other.coeff(i, j));
          m_matrix.coeffRef(j, j) = ei_real(other.coeff(j, j));
        }
        break;
    }
  }
}

template<typename MatrixType, unsigned int Mode>
template<typename Other> inline void Part<MatrixType, Mode>::operator+=(const Other& other)
{
  *this = m_matrix + other;
}

template<typename MatrixType, unsigned int Mode>
template<typename Other> inline void Part<MatrixType, Mode>::operator-=(const Other& other)
{
  *this = m_matrix - other;
}

template<typename MatrixType, unsigned int Mode>
inline void Part<MatrixType, Mode>::operator*=
(const typename ei_traits<MatrixType>::Scalar& other)
{
  *this = m_matrix * other;
}

template<typename MatrixType, unsigned int Mode>
inline void Part<MatrixType, Mode>::operator/=
(const typename ei_traits<MatrixType>::Scalar& other)
{
  *this = m_matrix / other;
}

template<typename MatrixType, unsigned int Mode>
inline void Part<MatrixType, Mode>::setConstant(const typename ei_traits<MatrixType>::Scalar& value)
{
  *this = MatrixType::constant(m_matrix.rows(), m_matrix.cols(), value);
}

template<typename MatrixType, unsigned int Mode>
inline void Part<MatrixType, Mode>::setZero()
{
  setConstant((typename ei_traits<MatrixType>::Scalar)(0));
}

template<typename MatrixType, unsigned int Mode>
inline void Part<MatrixType, Mode>::setOnes()
{
  setConstant((typename ei_traits<MatrixType>::Scalar)(1));
}

template<typename MatrixType, unsigned int Mode>
inline void Part<MatrixType, Mode>::setRandom()
{
  *this = MatrixType::random(m_matrix.rows(), m_matrix.cols());
}

/** \returns a lvalue pseudo-expression allowing to perform special operations on \c *this.
  *
  * The \a Mode parameter can have the following values: \c Upper, \c StrictlyUpper, \c Lower,
  * \c StrictlyLower, \c SelfAdjoint.
  *
  * Example: \include MatrixBase_part.cpp
  * Output: \verbinclude MatrixBase_part.out
  *
  * \sa class Part, MatrixBase::extract(), MatrixBase::marked()
  */
template<typename Derived>
template<unsigned int Mode>
inline Part<Derived, Mode> MatrixBase<Derived>::part()
{
  return Part<Derived, Mode>(derived());
}

#endif // EIGEN_PART_H
