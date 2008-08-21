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

#ifndef EIGEN_SPARSEMATRIX_H
#define EIGEN_SPARSEMATRIX_H

/** \class SparseMatrix
  *
  * \brief Sparse matrix
  *
  * \param _Scalar the scalar type, i.e. the type of the coefficients
  *
  * See http://www.netlib.org/linalg/html_templates/node91.html for details on the storage scheme.
  *
  */
template<typename _Scalar, int _Flags>
struct ei_traits<SparseMatrix<_Scalar, _Flags> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = SparseBit | _Flags,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = FullyCoherentAccessPattern
  };
};



template<typename _Scalar, int _Flags>
class SparseMatrix : public SparseMatrixBase<SparseMatrix<_Scalar, _Flags> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(SparseMatrix)

  protected:
  public:

    typedef SparseMatrixBase<SparseMatrix> SparseBase;
    enum {
      RowMajor = SparseBase::RowMajor
    };

    int m_outerSize;
    int m_innerSize;
    int* m_outerIndex;
    SparseArray<Scalar> m_data;


  public:

    inline int rows() const { return RowMajor ? m_outerSize : m_innerSize; }
    inline int cols() const { return RowMajor ? m_innerSize : m_outerSize; }
    inline int innerSize() const { return m_innerSize; }
    inline int outerSize() const { return m_outerSize; }
    inline int innerNonZeros(int j) const { return m_outerIndex[j+1]-m_outerIndex[j]; }

    inline Scalar coeff(int row, int col) const
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      int id = m_outerIndex[outer];
      int end = m_outerIndex[outer+1];
      // optimization: let's first check if it is the last coefficient
      // (very common in high level algorithms)
      if (end>0 && inner==m_data.index(end-1))
        return m_data.value(end-1);
      else if (id==end)
        return Scalar(0);
      const int* r = std::lower_bound(&m_data.index(id),&m_data.index(end),inner);
      return (*r==inner) ? m_data.value(r-&m_data.index(0)) : Scalar(0);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      int id = m_outerIndex[outer];
      int end = m_outerIndex[outer+1];
      ei_assert(end>=id && "you probably called coeffRef on a non finalized matrix");
      ei_assert(end>id && "coeffRef cannot be called on a zero coefficient");
      int* r = std::lower_bound(&m_data.index(id),&m_data.index(end),inner);
      ei_assert(*r==inner && "coeffRef cannot be called on a zero coefficient");
      return m_data.value(r-&m_data.index(0));
    }

  public:

    class InnerIterator;

    /** \returns the number of non zero coefficients */
    inline int nonZeros() const  { return m_data.size(); }

    inline void startFill(int reserveSize = 1000)
    {
      m_data.clear();
      m_data.reserve(reserveSize);
      for (int i=0; i<=m_outerSize; ++i)
        m_outerIndex[i] = 0;
    }

    inline Scalar& fill(int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      if (m_outerIndex[outer+1]==0)
      {
        int i=col;
        while (i>=0 && m_outerIndex[i]==0)
        {
          m_outerIndex[i] = m_data.size();
          --i;
        }
        m_outerIndex[outer+1] = m_outerIndex[outer];
      }
      assert(m_outerIndex[outer+1] == m_data.size());
      int id = m_outerIndex[outer+1];
      m_outerIndex[outer+1]++;

      m_data.append(0, inner);
      return m_data.value(id);
    }

    inline void endFill()
    {
      int size = m_data.size();
      int i = m_outerSize;
      // find the last filled column
      while (i>=0 && m_outerIndex[i]==0)
        --i;
      i++;
      while (i<=m_outerSize)
      {
        m_outerIndex[i] = size;
        ++i;
      }
    }

    void resize(int rows, int cols)
    {
      const int outerSize = RowMajor ? rows : cols;
      m_innerSize = RowMajor ? cols : rows;
      m_data.clear();
      if (m_outerSize != outerSize)
      {
        delete[] m_outerIndex;
        m_outerIndex = new int [outerSize+1];
        m_outerSize = outerSize;
      }
    }

    inline SparseMatrix(int rows, int cols)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0)
    {
      resize(rows, cols);
    }

    template<typename OtherDerived>
    inline SparseMatrix(const MatrixBase<OtherDerived>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0)
    {
      *this = other.derived();
    }

    inline void swap(SparseMatrix& other)
    {
      EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: swap\n");
      std::swap(m_outerIndex, other.m_outerIndex);
      std::swap(m_innerSize, other.m_innerSize);
      std::swap(m_outerSize, other.m_outerSize);
      m_data.swap(other.m_data);
    }

    inline SparseMatrix& operator=(const SparseMatrix& other)
    {
      if (other.isRValue())
      {
        swap(other.const_cast_derived());
      }
      else
      {
        resize(other.rows(), other.cols());
        for (int j=0; j<=m_outerSize; ++j)
          m_outerIndex[j] = other.m_outerIndex[j];
        m_data = other.m_data;
        return *this;
      }
    }

    template<typename OtherDerived>
    inline SparseMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return SparseMatrixBase<SparseMatrix>::operator=(other.derived());
    }

    friend std::ostream & operator << (std::ostream & s, const SparseMatrix& m)
    {
      EIGEN_DBG_SPARSE(
        s << "Nonzero entries:\n";
        for (uint i=0; i<m.nonZeros(); ++i)
        {
          s << "(" << m.m_data.value(i) << "," << m.m_data.index(i) << ") ";
        }
        s << std::endl;
        s << std::endl;
        s << "Column pointers:\n";
        for (uint i=0; i<m.cols(); ++i)
        {
          s << m.m_outerIndex[i] << " ";
        }
        s << std::endl;
        s << std::endl;
      );
      s << static_cast<const SparseMatrixBase<SparseMatrix>&>(m);
      return s;
    }

    /** Destructor */
    inline ~SparseMatrix()
    {
      delete[] m_outerIndex;
    }
};

template<typename Scalar, int _Flags>
class SparseMatrix<Scalar,_Flags>::InnerIterator
{
  public:
    InnerIterator(const SparseMatrix& mat, int outer)
      : m_matrix(mat), m_id(mat.m_outerIndex[outer]), m_start(m_id), m_end(mat.m_outerIndex[outer+1])
    {}

    InnerIterator& operator++() { m_id++; return *this; }

    Scalar value() { return m_matrix.m_data.value(m_id); }

    int index() const { return m_matrix.m_data.index(m_id); }

    operator bool() const { return (m_id < m_end) && (m_id>=m_start); }

  protected:
    const SparseMatrix& m_matrix;
    int m_id;
    const int m_start;
    const int m_end;
};

#endif // EIGEN_SPARSEMATRIX_H
