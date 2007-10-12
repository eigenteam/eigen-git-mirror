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

template<typename Lhs, typename Rhs> class Sum
  : public Object<typename Lhs::Scalar, Sum<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::ConstRef LhsRef;
    typedef typename Rhs::ConstRef RhsRef;
    friend class Object<Scalar, Sum>;
    
    static const int RowsAtCompileTime = Lhs::RowsAtCompileTime,
                     ColsAtCompileTime = Rhs::ColsAtCompileTime;

    Sum(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

    Sum(const Sum& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}
      
    EI_INHERIT_ASSIGNMENT_OPERATORS(Sum)

  private:
  
    const Sum& _ref() const { return *this; }
    const Sum& _constRef() const { return *this; }
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

template<typename Lhs, typename Rhs> class Difference
  : public Object<typename Lhs::Scalar, Difference<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::ConstRef LhsRef;
    typedef typename Rhs::ConstRef RhsRef;
    friend class Object<Scalar, Difference>;
    
    static const int RowsAtCompileTime = Lhs::RowsAtCompileTime,
                     ColsAtCompileTime = Rhs::ColsAtCompileTime;
    
    Difference(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }

    Difference(const Difference& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(Difference)

  private:
    const Difference& _ref() const { return *this; }
    const Difference& _constRef() const { return *this; }
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

template<int Index, int Size, typename Lhs, typename Rhs>
struct MatrixProductUnroller
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    MatrixProductUnroller<Index-1, Size, Lhs, Rhs>::run(row, col, lhs, rhs, res);
    res += lhs.read(row, Index) * rhs.read(Index, col);
  }
};

template<int Size, typename Lhs, typename Rhs>
struct MatrixProductUnroller<0, Size, Lhs, Rhs>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    res = lhs.read(row, 0) * rhs.read(0, col);
  }
};

template<int Index, typename Lhs, typename Rhs>
struct MatrixProductUnroller<Index, Dynamic, Lhs, Rhs>
{
  static void run(int row, int col, const Lhs& lhs, const Rhs& rhs,
                  typename Lhs::Scalar &res)
  {
    EI_UNUSED(row);
    EI_UNUSED(col);
    EI_UNUSED(lhs);
    EI_UNUSED(rhs);
    EI_UNUSED(res);
  }
};

template<typename Lhs, typename Rhs> class MatrixProduct
  : public Object<typename Lhs::Scalar, MatrixProduct<Lhs, Rhs> >
{
  public:
    typedef typename Lhs::Scalar Scalar;
    typedef typename Lhs::ConstRef LhsRef;
    typedef typename Rhs::ConstRef RhsRef;
    friend class Object<Scalar, MatrixProduct>;
    
    static const int RowsAtCompileTime = Lhs::RowsAtCompileTime,
                     ColsAtCompileTime = Rhs::ColsAtCompileTime;

    MatrixProduct(const LhsRef& lhs, const RhsRef& rhs)
      : m_lhs(lhs), m_rhs(rhs) 
    {
      assert(lhs.cols() == rhs.rows());
    }
    
    MatrixProduct(const MatrixProduct& other)
      : m_lhs(other.m_lhs), m_rhs(other.m_rhs) {}
    
    EI_INHERIT_ASSIGNMENT_OPERATORS(MatrixProduct)
    
  private:
    const MatrixProduct& _ref() const { return *this; }
    const MatrixProduct& _constRef() const { return *this; }
    int _rows() const { return m_lhs.rows(); }
    int _cols() const { return m_rhs.cols(); }
    
    Scalar _read(int row, int col) const
    {
      Scalar res;
      if(Lhs::ColsAtCompileTime != Dynamic && Lhs::ColsAtCompileTime <= 16)
        MatrixProductUnroller<Lhs::ColsAtCompileTime-1, Lhs::ColsAtCompileTime, LhsRef, RhsRef>
          ::run(row, col, m_lhs, m_rhs, res);
      else
      {
        res = m_lhs(row, 0) * m_rhs(0, col);
        for(int i = 1; i < m_lhs.cols(); i++)
          res += m_lhs(row, i) * m_rhs(i, col);
      }
      return res;
    }
    
  protected:
    const LhsRef m_lhs;
    const RhsRef m_rhs;
};

template<typename Scalar, typename Derived1, typename Derived2>
Sum<Derived1, Derived2>
operator+(const Object<Scalar, Derived1> &mat1, const Object<Scalar, Derived2> &mat2)
{
  return Sum<Derived1, Derived2>(mat1.constRef(), mat2.constRef());
}

template<typename Scalar, typename Derived1, typename Derived2>
Difference<Derived1, Derived2>
operator-(const Object<Scalar, Derived1> &mat1, const Object<Scalar, Derived2> &mat2)
{
  return Difference<Derived1, Derived2>(mat1.constRef(), mat2.constRef());
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
MatrixProduct<Derived, OtherDerived>
Object<Scalar, Derived>::lazyMul(const Object<Scalar, OtherDerived> &other) const
{
  return MatrixProduct<Derived, OtherDerived>(constRef(), other.constRef());
}

template<typename Scalar, typename Derived1, typename Derived2>
Eval<MatrixProduct<Derived1, Derived2> >
operator*(const Object<Scalar, Derived1> &mat1, const Object<Scalar, Derived2> &mat2)
{
  return mat1.lazyMul(mat2).eval();
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
Object<Scalar, Derived>::operator+=(const Object<Scalar, OtherDerived>& other)
{
  return *this = *this + other;
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
Object<Scalar, Derived>::operator-=(const Object<Scalar, OtherDerived> &other)
{
  return *this = *this - other;
}

template<typename Scalar, typename Derived>
template<typename OtherDerived>
Derived &
Object<Scalar, Derived>::operator*=(const Object<Scalar, OtherDerived> &other)
{
  return *this = *this * other;
}

#endif // EI_MATRIXOPS_H
