// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_SPARSEMATRIXBASE_H
#define EIGEN_SPARSEMATRIXBASE_H

template<typename Derived>
class SparseMatrixBase : public MatrixBase<Derived>
{
  public:

    typedef MatrixBase<Derived> Base;
    typedef typename Base::Scalar Scalar;
    enum {
      Flags = Base::Flags,
      RowMajor = ei_traits<Derived>::Flags&RowMajorBit ? 1 : 0
    };

    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }
    inline Derived& const_cast_derived() const
    { return *static_cast<Derived*>(const_cast<SparseMatrixBase*>(this)); }

    SparseMatrixBase()
      : m_isRValue(false)
    {}

    bool isRValue() const { return m_isRValue; }
    Derived& markAsRValue() { m_isRValue = true; return derived(); }

    inline Derived& operator=(const Derived& other)
    {
//       std::cout << "Derived& operator=(const Derived& other)\n";
      if (other.isRValue())
        derived().swap(other.const_cast_derived());
      else
        this->operator=<Derived>(other);
      return derived();
    }

    template<typename OtherDerived>
    inline Derived& operator=(const MatrixBase<OtherDerived>& other)
    {
//       std::cout << "Derived& operator=(const MatrixBase<OtherDerived>& other)\n";
      const bool transpose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
//       std::cout << "eval transpose = " << transpose << "\n";
      const int outerSize = other.outerSize();
      typedef typename ei_meta_if<transpose, LinkedVectorMatrix<Scalar,Flags&RowMajorBit>, Derived>::ret TempType;
      TempType temp(other.rows(), other.cols());

      temp.startFill(std::max(this->rows(),this->cols())*2);
      for (int j=0; j<outerSize; ++j)
      {
        for (typename OtherDerived::InnerIterator it(other.derived(), j); it; ++it)
        {
          Scalar v = it.value();
          if (v!=Scalar(0))
          {
            if (OtherDerived::Flags & RowMajorBit) temp.fill(j,it.index()) = v;
            else temp.fill(it.index(),j) = v;
          }
        }
      }
      temp.endFill();

      derived() = temp.markAsRValue();
      return derived();
    }

    template<typename OtherDerived>
    inline Derived& operator=(const SparseMatrixBase<OtherDerived>& other)
    {
//       std::cout << typeid(OtherDerived).name() << "\n";
//       std::cout << Flags << " " << OtherDerived::Flags << "\n";
      const bool transpose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
//       std::cout << "eval transpose = " << transpose << "\n";
      const int outerSize = (int(OtherDerived::Flags) & RowMajorBit) ? other.rows() : other.cols();
      if ((!transpose) && other.isRValue())
      {
        // eval without temporary
        derived().resize(other.rows(), other.cols());
        derived().startFill(std::max(this->rows(),this->cols())*2);
        for (int j=0; j<outerSize; ++j)
        {
          for (typename OtherDerived::InnerIterator it(other.derived(), j); it; ++it)
          {
            Scalar v = it.value();
            if (v!=Scalar(0))
            {
              if (RowMajor) derived().fill(j,it.index()) = v;
              else derived().fill(it.index(),j) = v;
            }
          }
        }
        derived().endFill();
      }
      else
        this->operator=<OtherDerived>(static_cast<const MatrixBase<OtherDerived>&>(other));
      return derived();
    }

    template<typename Lhs, typename Rhs>
    inline Derived& operator=(const Product<Lhs,Rhs,SparseProduct>& product);

    friend std::ostream & operator << (std::ostream & s, const SparseMatrixBase& m)
    {
      if (Flags&RowMajorBit)
      {
        for (int row=0; row<m.outerSize(); ++row)
        {
          int col = 0;
          for (typename Derived::InnerIterator it(m.derived(), row); it; ++it)
          {
            for ( ; col<it.index(); ++col)
              s << "0 ";
            s << it.value() << " ";
            ++col;
          }
          for ( ; col<m.cols(); ++col)
            s << "0 ";
          s << std::endl;
        }
      }
      else
      {
        if (m.cols() == 1) {
          int row = 0;
          for (typename Derived::InnerIterator it(m.derived(), 0); it; ++it)
          {
            for ( ; row<it.index(); ++row)
              s << "0" << std::endl;
            s << it.value() << std::endl;
            ++row;
          }
          for ( ; row<m.rows(); ++row)
            s << "0" << std::endl;
        }
        else
        {
          LinkedVectorMatrix<Scalar, RowMajorBit> trans = m.derived();
          s << trans;
        }
      }
      return s;
    }

  protected:

    bool m_isRValue;
};

#endif // EIGEN_SPARSEMATRIXBASE_H
