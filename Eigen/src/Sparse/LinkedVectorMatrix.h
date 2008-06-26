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

#ifndef EIGEN_LINKEDVECTORMATRIX_H
#define EIGEN_LINKEDVECTORMATRIX_H

template<typename _Scalar, int _Flags>
struct ei_traits<LinkedVectorMatrix<_Scalar,_Flags> >
{
  typedef _Scalar Scalar;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = _Flags,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = InnerCoherentAccessPattern
  };
};

template<typename Element, int BlockSize = 8>
struct LinkedVector
{
  LinkedVector() : size(0), next(0), prev(0) {}
  Element data[BlockSize];
  LinkedVector* next;
  LinkedVector* prev;
  int size;
  bool isFull() const { return size==BlockSize; }
};

template<typename _Scalar, int _Flags>
class LinkedVectorMatrix : public SparseMatrixBase<LinkedVectorMatrix<_Scalar,_Flags> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(LinkedVectorMatrix)
    class InnerIterator;
  protected:

    enum {
      RowMajor = Flags&RowMajorBit ? 1 : 0
    };

    struct ValueIndex
    {
      ValueIndex() : value(0), index(0) {}
      ValueIndex(Scalar v, int i) : value(v), index(i) {}
      Scalar value;
      int index;
    };
    typedef LinkedVector<ValueIndex,8> LinkedVectorBlock;

    inline int find(LinkedVectorBlock** _el, int id)
    {
      LinkedVectorBlock* el = *_el;
      while (el && el->data[el->size-1].index<id)
        el = el->next;
      *_el = el;
      if (el)
      {
        // binary search
        int maxI = el->size-1;
        int minI = 0;
        int i = el->size/2;
        const ValueIndex* data = el->data;
        while (data[i].index!=id)
        {
          if (data[i].index<id)
          {
            minI = i+1;
            i = (maxI + minI)+2;
          }
          else
          {
            maxI = i-1;
            i = (maxI + minI)+2;
          }
          if (minI>=maxI)
            return -1;
        }
        if (data[i].index==id)
          return i;
      }
      return -1;
    }

  public:
    inline int rows() const { return RowMajor ? m_data.size() : m_innerSize; }
    inline int cols() const { return RowMajor ? m_innerSize : m_data.size(); }

    inline const Scalar& coeff(int row, int col) const
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      LinkedVectorBlock* el = m_data[outer];
      int id = find(&el, inner);
      if (id<0)
        return Scalar(0);
      return el->data[id].value;
    }

    inline Scalar& coeffRef(int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;

      LinkedVectorBlock* el = m_data[outer];
      int id = find(&el, inner);
      ei_assert(id>=0);
//       if (id<0)
//         return Scalar(0);
      return el->data[id].value;
    }

  public:

    inline void startFill(int reserveSize = 1000)
    {
      clear();
      for (int i=0; i<m_data.size(); ++i)
        m_ends[i] = m_data[i] = 0;
    }

    inline Scalar& fill(int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;
      if (m_ends[outer]==0)
      {
        m_data[outer] = m_ends[outer] = new LinkedVectorBlock();
      }
      else
      {
        ei_assert(m_ends[outer]->data[m_ends[outer]->size-1].index < inner);
        if (m_ends[outer]->isFull())
        {

          LinkedVectorBlock* el = new LinkedVectorBlock();
          m_ends[outer]->next = el;
          el->prev = m_ends[outer];
          m_ends[outer] = el;
        }
      }
      m_ends[outer]->data[m_ends[outer]->size].index = inner;
      return m_ends[outer]->data[m_ends[outer]->size++].value;
    }

    inline void endFill()
    {
    }

    ~LinkedVectorMatrix()
    {
      if (this->isNotShared())
        clear();
    }

    void clear()
    {
      for (int i=0; i<m_data.size(); ++i)
      {
        LinkedVectorBlock* el = m_data[i];
        while (el)
        {
          LinkedVectorBlock* tmp = el;
          el = el->next;
          delete tmp;
        }
      }
    }

    void resize(int rows, int cols)
    {
      const int outers = RowMajor ? rows : cols;
      const int inners = RowMajor ? cols : rows;

      if (this->outerSize() != outers)
      {
        clear();
        m_data.resize(outers);
        m_ends.resize(outers);
        for (int i=0; i<m_data.size(); ++i)
          m_ends[i] = m_data[i] = 0;
      }
      m_innerSize = inners;
    }

    inline LinkedVectorMatrix(int rows, int cols)
      : m_innerSize(0)
    {
      resize(rows, cols);
    }

    template<typename OtherDerived>
    inline LinkedVectorMatrix(const MatrixBase<OtherDerived>& other)
      : m_innerSize(0)
    {
      *this = other.derived();
    }

    inline void shallowCopy(const LinkedVectorMatrix& other)
    {
      EIGEN_DBG_SPARSE(std::cout << "LinkedVectorMatrix:: shallowCopy\n");
      resize(other.rows(), other.cols());
      for (int j=0; j<this->outerSize(); ++j)
      {
        m_data[j] = other.m_data[j];
        m_ends[j] = other.m_ends[j];
      }
      other.markAsCopied();
    }

    inline LinkedVectorMatrix& operator=(const LinkedVectorMatrix& other)
    {
      if (other.isRValue())
      {
        shallowCopy(other);
        return *this;
      }
      else
      {
        // TODO implement a specialized deep copy here
        return operator=<LinkedVectorMatrix>(other);
      }
    }

    template<typename OtherDerived>
    inline LinkedVectorMatrix& operator=(const MatrixBase<OtherDerived>& other)
    {
      return SparseMatrixBase<LinkedVectorMatrix>::operator=(other.derived());
    }

  protected:

    std::vector<LinkedVectorBlock*> m_data;
    std::vector<LinkedVectorBlock*> m_ends;
    int m_innerSize;

};


template<typename Scalar, int _Flags>
class LinkedVectorMatrix<Scalar,_Flags>::InnerIterator
{
  public:

    InnerIterator(const LinkedVectorMatrix& mat, int col)
      : m_matrix(mat), m_el(mat.m_data[col]), m_it(0)
    {}

    InnerIterator& operator++() { if (m_it<m_el->size) m_it++; else {m_el = m_el->next; m_it=0;}; return *this; }

    Scalar value() { return m_el->data[m_it].value; }

    int index() const { return m_el->data[m_it].index; }

    operator bool() const { return m_el && (m_el->next || m_it<m_el->size); }

  protected:
    const LinkedVectorMatrix& m_matrix;
    LinkedVectorBlock* m_el;
    int m_it;
};

#endif // EIGEN_LINKEDVECTORMATRIX_H
