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
    Flags = _Flags,
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

    int* m_colPtrs;
    SparseArray<Scalar> m_data;
    int m_rows;
    int m_cols;

  public:

    inline int rows() const { return m_rows; }
    inline int cols() const { return m_cols; }

    inline const Scalar& coeff(int row, int col) const
    {
      int id = m_colPtrs[col];
      int end = m_colPtrs[col+1];
      while (id<end && m_data.index(id)!=row)
      {
        ++id;
      }
      if (id==end)
        return 0;
      return m_data.value(id);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      int id = m_colPtrs[col];
      int end = m_colPtrs[col+1];
      while (id<end && m_data.index(id)!=row)
      {
        ++id;
      }
      ei_assert(id!=end);
      return m_data.value(id);
    }

  public:

    class InnerIterator;

    /** \returns the number of non zero coefficients */
    inline int nonZeros() const  { return m_data.size(); }

    inline void startFill(int reserveSize = 1000)
    {
      m_data.clear();
      m_data.reserve(reserveSize);
      for (int i=0; i<=m_cols; ++i)
        m_colPtrs[i] = 0;
    }

    inline Scalar& fill(int row, int col)
    {
      if (m_colPtrs[col+1]==0)
      {
        int i=col;
        while (i>=0 && m_colPtrs[i]==0)
        {
          m_colPtrs[i] = m_data.size();
          --i;
        }
        m_colPtrs[col+1] = m_colPtrs[col];
      }
      assert(m_colPtrs[col+1] == m_data.size());
      int id = m_colPtrs[col+1];
      m_colPtrs[col+1]++;

      m_data.append(0, row);
      return m_data.value(id);
    }

    inline void endFill()
    {
      int size = m_data.size();
      int i = m_cols;
      // find the last filled column
      while (i>=0 && m_colPtrs[i]==0)
        --i;
      i++;
      while (i<=m_cols)
      {
        m_colPtrs[i] = size;
        ++i;
      }
    }

    void resize(int rows, int cols)
    {
      if (m_cols != cols)
      {
        delete[] m_colPtrs;
        m_colPtrs = new int [cols+1];
        m_rows = rows;
        m_cols = cols;
      }
    }

    inline SparseMatrix(int rows, int cols)
      : m_rows(0), m_cols(0), m_colPtrs(0)
    {
      resize(rows, cols);
    }

    inline void shallowCopy(const SparseMatrix& other)
    {
      EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: shallowCopy\n");
      delete[] m_colPtrs;
      m_colPtrs = 0;
      m_rows = other.rows();
      m_cols = other.cols();
      m_colPtrs = other.m_colPtrs;
      m_data.shallowCopy(other.m_data);
      other.markAsCopied();
    }

    inline SparseMatrix& operator=(const SparseMatrix& other)
    {
      if (other.isRValue())
      {
        shallowCopy(other);
      }
      else
      {
        resize(other.rows(), other.cols());
        for (int col=0; col<=cols(); ++col)
          m_colPtrs[col] = other.m_colPtrs[col];
        m_data = other.m_data;
        return *this;
      }
    }

    template<typename OtherDerived>
    inline SparseMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return SparseMatrixBase<SparseMatrix>::operator=(other);
    }

    template<typename OtherDerived>
    SparseMatrix<Scalar> operator*(const MatrixBase<OtherDerived>& other)
    {
      SparseMatrix<Scalar> res(rows(), other.cols());
      ei_sparse_product<SparseMatrix,OtherDerived>(*this,other.derived(),res);
      return res;
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
          s << m.m_colPtrs[i] << " ";
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
      if (this->isNotShared())
        delete[] m_colPtrs;
    }
};

template<typename Scalar, int _Flags>
class SparseMatrix<Scalar,_Flags>::InnerIterator
{
  public:
    InnerIterator(const SparseMatrix& mat, int col)
      : m_matrix(mat), m_id(mat.m_colPtrs[col]), m_start(m_id), m_end(mat.m_colPtrs[col+1])
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
