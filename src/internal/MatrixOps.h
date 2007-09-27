// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EI_MATRIXOPS_H
#define EI_MATRIXOPS_H

template<typename Lhs, typename Rhs> class EiSum
  : public EiObject<typename Lhs::Scalar, EiSum<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::Ref LhsRef;
    typedef typename Rhs::Ref RhsRef;
    friend class EiObject<Scalar, EiSum>;
    typedef EiSum Ref;
    
    static const int RowsAtCompileTime = Lhs::RowsAtCompileTime,
                     ColsAtCompileTime = Rhs::ColsAtCompileTime;

    EiSum(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

    EiSum(const EiSum& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(EiSum)

  private:
  
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_lhs.cols(); }

    Scalar _read(int row, int col) const
    {
      return m_lhs.read(row, col) + m_rhs.read(row, col);
    }
    
  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

template<typename Lhs, typename Rhs> class EiDifference
  : public EiObject<typename Lhs::Scalar, EiDifference<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::Ref LhsRef;
    typedef typename Rhs::Ref RhsRef;
    friend class EiObject<Scalar, EiDifference>;
    typedef EiDifference Ref;
    
    static const int RowsAtCompileTime = Lhs::RowsAtCompileTime,
                     ColsAtCompileTime = Rhs::ColsAtCompileTime;
    
    EiDifference(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

    EiDifference(const EiDifference& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(EiDifference)

  private:
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_lhs.cols(); }

    Scalar _read(int row, int col) const
    {
      return m_lhs.read(row, col) - m_rhs.read(row, col);
    }
    
  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

template<typename Lhs, typename Rhs> class EiMatrixProduct
  : public EiObject<typename Lhs::Scalar, EiMatrixProduct<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::Ref LhsRef;
    typedef typename Rhs::Ref RhsRef;
    friend class EiObject<Scalar, EiMatrixProduct>;
    typedef EiMatrixProduct Ref;
    
    static const int RowsAtCompileTime = Lhs::RowsAtCompileTime,
                     ColsAtCompileTime = Rhs::ColsAtCompileTime;

    EiMatrixProduct(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs) 
    {
      assert(lhs.cols() == rhs.rows());
    }
    
    EiMatrixProduct(const EiMatrixProduct& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}
    
    EI_INHERIT_ASSIGNMENT_OPERATORS(EiMatrixProduct)
    
  private:
    const Ref& _ref() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_rhs.cols(); }
    
    Scalar _read(int row, int col) const
    {
      Scalar x = static_cast<Scalar>(0);
      for(int i = 0; i < m_lhs.cols(); i++)
        x += m_lhs.read(row, i) * m_rhs.read(i, col);
      return x;
    }
    
  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

template<typename Scalar, typename Derived1, typename Derived2>
EiSum<Derived1, Derived2>
operator+(const EiObject<Scalar, Derived1> &mat1, const EiObject<Scalar, Derived2> &mat2)
{
  return EiSum<Derived1, Derived2>(mat1.ref(), mat2.ref());
}

template<typename Scalar, typename Derived1, typename Derived2>
EiDifference<Derived1, Derived2>
operator-(const EiObject<Scalar, Derived1> &mat1, const EiObject<Scalar, Derived2> &mat2)
{
  return EiDifference<Derived1, Derived2>(mat1.ref(), mat2.ref());
}

template<typename Scalar, typename Derived1, typename Derived2>
EiMatrixProduct<Derived1, Derived2>
operator*(const EiObject<Scalar, Derived1> &mat1, const EiObject<Scalar, Derived2> &mat2)
{
  return EiMatrixProduct<Derived1, Derived2>(mat1.ref(), mat2.ref());
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
EiObject<Scalar, Derived>::operator+=(const EiObject<Scalar, OtherDerived>& other)
{
  *this = *this + other;
  return *static_cast<Derived*>(this);
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
EiObject<Scalar, Derived>::operator-=(const EiObject<Scalar, OtherDerived> &other)
{
  *this = *this - other;
  return *static_cast<Derived*>(this);
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
EiObject<Scalar, Derived>::operator*=(const EiObject<Scalar, OtherDerived> &other)
{
  *this = *this * other;
  return *static_cast<Derived*>(this);
}

#endif // EI_MATRIXOPS_H
