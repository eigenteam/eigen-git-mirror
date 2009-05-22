// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

#ifndef EIGEN_HASHMATRIX_H
#define EIGEN_HASHMATRIX_H

template<typename _Scalar, int _Flags>
struct ei_traits<HashMatrix<_Scalar, _Flags> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = SparseBit | _Flags,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = RandomAccessPattern
  };
};

// TODO reimplement this class using custom linked lists
template<typename _Scalar, int _Flags>
class HashMatrix
  : public SparseMatrixBase<HashMatrix<_Scalar, _Flags> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(HashMatrix)
    class InnerIterator;
  protected:

    typedef typename std::map<int, Scalar>::iterator MapIterator;
    typedef typename std::map<int, Scalar>::const_iterator ConstMapIterator;

  public:
    inline int rows() const { return m_innerSize; }
    inline int cols() const { return m_data.size(); }

    inline const Scalar& coeff(int row, int col) const
    {
      const MapIterator it = m_data[col].find(row);
      if (it!=m_data[col].end())
        return Scalar(0);
      return it->second;
    }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_data[col][row];
    }

  public:

    inline void startFill(int /*reserveSize = 1000 --- currently unused, don't generate a warning*/) {}

    inline Scalar& fill(int row, int col) { return coeffRef(row, col); }

    inline void endFill() {}

    ~HashMatrix()
    {}

    inline void shallowCopy(const HashMatrix& other)
    {
      EIGEN_DBG_SPARSE(std::cout << "HashMatrix:: shallowCopy\n");
      // FIXME implement a true shallow copy !!
      resize(other.rows(), other.cols());
      for (int j=0; j<this->outerSize(); ++j)
        m_data[j] = other.m_data[j];
    }

    void resize(int _rows, int _cols)
    {
      if (cols() != _cols)
      {
        m_data.resize(_cols);
      }
      m_innerSize = _rows;
    }

    inline HashMatrix(int rows, int cols)
      : m_innerSize(0)
    {
      resize(rows, cols);
    }

    template<typename OtherDerived>
    inline HashMatrix(const MatrixBase<OtherDerived>& other)
      : m_innerSize(0)
    {
      *this = other.derived();
    }

    inline HashMatrix& operator=(const HashMatrix& other)
    {
      if (other.isRValue())
      {
        shallowCopy(other);
      }
      else
      {
        resize(other.rows(), other.cols());
        for (int col=0; col<cols(); ++col)
          m_data[col] = other.m_data[col];
      }
      return *this;
    }

    template<typename OtherDerived>
    inline HashMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return SparseMatrixBase<HashMatrix>::operator=(other);
    }

  protected:

    std::vector<std::map<int, Scalar> > m_data;
    int m_innerSize;

};

template<typename Scalar, int _Flags>
class HashMatrix<Scalar,_Flags>::InnerIterator
{
  public:

    InnerIterator(const HashMatrix& mat, int col)
      : m_matrix(mat), m_it(mat.m_data[col].begin()), m_end(mat.m_data[col].end())
    {}

    InnerIterator& operator++() { m_it++; return *this; }

    Scalar value() { return m_it->second; }

    int index() const { return m_it->first; }

    operator bool() const { return m_it!=m_end; }

  protected:
    const HashMatrix& m_matrix;
    typename HashMatrix::ConstMapIterator m_it;
    typename HashMatrix::ConstMapIterator m_end;
};

#endif // EIGEN_HASHMATRIX_H
