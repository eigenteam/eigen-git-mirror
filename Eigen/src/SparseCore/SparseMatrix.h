// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
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

/** \ingroup SparseCore_Module
  *
  * \class SparseMatrix
  *
  * \brief A versatible sparse matrix representation
  *
  * This class implements a more versatile variants of the common \em compressed row/column storage format.
  * Each colmun's (resp. row) non zeros are stored as a pair of value with associated row (resp. colmiun) index.
  * All the non zeros are stored in a single large buffer. Unlike the \em compressed format, there might be extra
  * space inbetween the nonzeros of two successive colmuns (resp. rows) such that insertion of new non-zero
  * can be done with limited memory reallocation and copies.
  *
  * A call to the function makeCompressed() turns the matrix into the standard \em compressed format
  * compatible with many library.
  *
  * More details on this storage sceheme are given in the \ref TutorialSparse "manual pages".
  *
  * \tparam _Scalar the scalar type, i.e. the type of the coefficients
  * \tparam _Options Union of bit flags controlling the storage scheme. Currently the only possibility
  *                 is RowMajor. The default is 0 which means column-major.
  * \tparam _Index the type of the indices. Default is \c int.
  *
  * This class can be extended with the help of the plugin mechanism described on the page
  * \ref TopicCustomizingEigen by defining the preprocessor symbol \c EIGEN_SPARSEMATRIX_PLUGIN.
  */

namespace internal {
template<typename _Scalar, int _Options, typename _Index>
struct traits<SparseMatrix<_Scalar, _Options, _Index> >
{
  typedef _Scalar Scalar;
  typedef _Index Index;
  typedef Sparse StorageKind;
  typedef MatrixXpr XprKind;
  enum {
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = Dynamic,
    MaxColsAtCompileTime = Dynamic,
    Flags = _Options | NestByRefBit | LvalueBit,
    CoeffReadCost = NumTraits<Scalar>::ReadCost,
    SupportedAccessPatterns = InnerRandomAccessPattern
  };
};

} // end namespace internal

template<typename _Scalar, int _Options, typename _Index>
class SparseMatrix
  : public SparseMatrixBase<SparseMatrix<_Scalar, _Options, _Index> >
{
  public:
    EIGEN_SPARSE_PUBLIC_INTERFACE(SparseMatrix)
    EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(SparseMatrix, +=)
    EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(SparseMatrix, -=)

    typedef MappedSparseMatrix<Scalar,Flags> Map;
    using Base::IsRowMajor;
    typedef CompressedStorage<Scalar,Index> Storage;
    enum {
      Options = _Options
    };

  protected:

    typedef SparseMatrix<Scalar,(Flags&~RowMajorBit)|(IsRowMajor?RowMajorBit:0)> TransposedSparseMatrix;

    Index m_outerSize;
    Index m_innerSize;
    Index* m_outerIndex;
    Index* m_innerNonZeros;     // optional, if null then the data is compressed
    CompressedStorage<Scalar,Index> m_data;
    
    Eigen::Map<Matrix<Index,Dynamic,1> > innerNonZeros() { return Eigen::Map<Matrix<Index,Dynamic,1> >(m_innerNonZeros, m_innerNonZeros?m_outerSize:0); }
    const  Eigen::Map<const Matrix<Index,Dynamic,1> > innerNonZeros() const { return Eigen::Map<const Matrix<Index,Dynamic,1> >(m_innerNonZeros, m_innerNonZeros?m_outerSize:0); }

  public:
    
    inline bool compressed() const { return m_innerNonZeros==0; }

    inline Index rows() const { return IsRowMajor ? m_outerSize : m_innerSize; }
    inline Index cols() const { return IsRowMajor ? m_innerSize : m_outerSize; }

    inline Index innerSize() const { return m_innerSize; }
    inline Index outerSize() const { return m_outerSize; }
    /** \returns the number of non zeros in the inner vector \a j
      */
    inline Index innerNonZeros(Index j) const
    {
      return m_innerNonZeros ? m_innerNonZeros[j] : m_outerIndex[j+1]-m_outerIndex[j];
    }
    
    /** \internal
      * \returns a const pointer to the array of values */
    inline const Scalar* _valuePtr() const { return &m_data.value(0); }
    /** \internal
      * \returns a non-const pointer to the array of values */
    inline Scalar* _valuePtr() { return &m_data.value(0); }

    /** \internal
      * \returns a const pointer to the array of inner indices */
    inline const Index* _innerIndexPtr() const { return &m_data.index(0); }
    /** \internal
      * \returns a non-const pointer to the array of inner indices */
    inline Index* _innerIndexPtr() { return &m_data.index(0); }

    /** \internal
      * \returns a const pointer to the array of the starting positions of the inner vectors */
    inline const Index* _outerIndexPtr() const { return m_outerIndex; }
    /** \internal
      * \returns a non-const pointer to the array of the starting positions of the inner vectors */
    inline Index* _outerIndexPtr() { return m_outerIndex; }

    inline Storage& data() { return m_data; }
    inline const Storage& data() const { return m_data; }

    /** \returns the value of the matrix at position \a i, \a j
      * This function returns Scalar(0) if the element is an explicit \em zero */
    inline Scalar coeff(Index row, Index col) const
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      return m_data.atInRange(m_outerIndex[outer], end, inner);
    }

    /** \returns a non-const reference to the value of the matrix at position \a i, \a j
      * The element \b have to be a non-zero element. */
    inline Scalar& coeffRef(Index row, Index col)
    {
      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index start = m_outerIndex[outer];
      Index end = m_innerNonZeros ? m_outerIndex[outer] + m_innerNonZeros[outer] : m_outerIndex[outer+1];
      eigen_assert(end>=start && "you probably called coeffRef on a non finalized matrix");
      eigen_assert(end>start && "coeffRef cannot be called on a zero coefficient");
      const Index p = m_data.searchLowerIndex(start,end-1,inner);
      eigen_assert((p<end) && (m_data.index(p)==inner) && "coeffRef cannot be called on a zero coefficient");
      return m_data.value(p);
    }

  public:

    class InnerIterator;

    /** Removes all non zeros but keep allocated memory */
    inline void setZero()
    {
      m_data.clear();
      memset(m_outerIndex, 0, (m_outerSize+1)*sizeof(Index));
      if(m_innerNonZeros)
        memset(m_innerNonZeros, 0, (m_outerSize)*sizeof(Index));
    }

    /** \returns the number of non zero coefficients */
    inline Index nonZeros() const
    {
      if(m_innerNonZeros)
        return innerNonZeros().sum();
      return static_cast<Index>(m_data.size());
    }

    /** Preallocates \a reserveSize non zeros.
      *
      * Precondition: the matrix must be in compressed mode. */
    inline void reserve(Index reserveSize)
    {
      eigen_assert(compressed() && "This function does not make sense in non compressed mode.");
      m_data.reserve(reserveSize);
    }
    
    #ifdef EIGEN_PARSED_BY_DOXYGEN
    /** Preallocates \a reserveSize[\c j] non zeros for each column (resp. row) \c j.
      *
      * This function turns the matrix in non-compressed() mode */
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes);
    #else
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes, const typename SizesType::value_type& enableif = typename SizesType::value_type())
    {
      EIGEN_UNUSED_VARIABLE(enableif);
      reserveInnerVectors(reserveSizes);
    }
    template<class SizesType>
    inline void reserve(const SizesType& reserveSizes, const typename SizesType::Scalar& enableif = typename SizesType::Scalar())
    {
      EIGEN_UNUSED_VARIABLE(enableif);
      reserveInnerVectors(reserveSizes);
    }
    #endif // EIGEN_PARSED_BY_DOXYGEN
  protected:
    template<class SizesType>
    inline void reserveInnerVectors(const SizesType& reserveSizes)
    {
      
      if(compressed())
      {
        std::size_t totalReserveSize = 0;
        // turn the matrix into non-compressed mode
        m_innerNonZeros = new Index[m_outerSize];
        
        // temporarily use m_innerSizes to hold the new starting points.
        Index* newOuterIndex = m_innerNonZeros;
        
        Index count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          count += reserveSizes[j] + (m_outerIndex[j+1]-m_outerIndex[j]);
          totalReserveSize += reserveSizes[j];
        }
        m_data.reserve(totalReserveSize);
        std::ptrdiff_t previousOuterIndex = m_outerIndex[m_outerSize];
        for(std::ptrdiff_t j=m_outerSize-1; j>=0; --j)
        {
          ptrdiff_t innerNNZ = previousOuterIndex - m_outerIndex[j];
          for(std::ptrdiff_t i=innerNNZ-1; i>=0; --i)
          {
            m_data.index(newOuterIndex[j]+i) = m_data.index(m_outerIndex[j]+i);
            m_data.value(newOuterIndex[j]+i) = m_data.value(m_outerIndex[j]+i);
          }
          previousOuterIndex = m_outerIndex[j];
          m_outerIndex[j] = newOuterIndex[j];
          m_innerNonZeros[j] = innerNNZ;
        }
        m_outerIndex[m_outerSize] = m_outerIndex[m_outerSize-1] + m_innerNonZeros[m_outerSize-1] + reserveSizes[m_outerSize-1];
        
        m_data.resize(m_outerIndex[m_outerSize]);
      }
      else
      {
        Index* newOuterIndex = new Index[m_outerSize+1];
        Index count = 0;
        for(Index j=0; j<m_outerSize; ++j)
        {
          newOuterIndex[j] = count;
          Index alreadyReserved = (m_outerIndex[j+1]-m_outerIndex[j]) - m_innerNonZeros[j];
          Index toReserve = std::max<std::ptrdiff_t>(reserveSizes[j], alreadyReserved);
          count += toReserve + m_innerNonZeros[j];
        }
        newOuterIndex[m_outerSize] = count;
        
        m_data.resize(count);
        for(ptrdiff_t j=m_outerSize-1; j>=0; --j)
        {
          std::ptrdiff_t offset = newOuterIndex[j] - m_outerIndex[j];
          if(offset>0)
          {
            std::ptrdiff_t innerNNZ = m_innerNonZeros[j];
            for(std::ptrdiff_t i=innerNNZ-1; i>=0; --i)
            {
              m_data.index(newOuterIndex[j]+i) = m_data.index(m_outerIndex[j]+i);
              m_data.value(newOuterIndex[j]+i) = m_data.value(m_outerIndex[j]+i);
            }
          }
        }
        
        std::swap(m_outerIndex, newOuterIndex);
        delete[] newOuterIndex;
      }
      
    }
  public:

    //--- low level purely coherent filling ---

    /** \internal
      * \returns a reference to the non zero coefficient at position \a row, \a col assuming that:
      * - the nonzero does not already exist
      * - the new coefficient is the last one according to the storage order
      *
      * Before filling a given inner vector you must call the statVec(Index) function.
      *
      * After an insertion session, you should call the finalize() function.
      *
      * \sa insert, insertBackByOuterInner, startVec */
    inline Scalar& insertBack(Index row, Index col)
    {
      return insertBackByOuterInner(IsRowMajor?row:col, IsRowMajor?col:row);
    }

    /** \internal
      * \sa insertBack, startVec */
    inline Scalar& insertBackByOuterInner(Index outer, Index inner)
    {
      eigen_assert(size_t(m_outerIndex[outer+1]) == m_data.size() && "Invalid ordered insertion (invalid outer index)");
      eigen_assert( (m_outerIndex[outer+1]-m_outerIndex[outer]==0 || m_data.index(m_data.size()-1)<inner) && "Invalid ordered insertion (invalid inner index)");
      Index p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(0, inner);
      return m_data.value(p);
    }

    /** \internal
      * \warning use it only if you know what you are doing */
    inline Scalar& insertBackByOuterInnerUnordered(Index outer, Index inner)
    {
      Index p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];
      m_data.append(0, inner);
      return m_data.value(p);
    }

    /** \internal
      * \sa insertBack, insertBackByOuterInner */
    inline void startVec(Index outer)
    {
      eigen_assert(m_outerIndex[outer]==int(m_data.size()) && "You must call startVec for each inner vector sequentially");
      eigen_assert(m_outerIndex[outer+1]==0 && "You must call startVec for each inner vector sequentially");
      m_outerIndex[outer+1] = m_outerIndex[outer];
    }

    //---

    /** \returns a reference to a novel non zero coefficient with coordinates \a row x \a col.
      * The non zero coefficient must \b not already exist.
      *
      * \warning This function can be extremely slow if the non zero coefficients
      * are not inserted in a coherent order.
      *
      * After an insertion session, you should call the finalize() function.
      */
    EIGEN_DONT_INLINE Scalar& insert(Index row, Index col)
    {
      if(compressed())
        return insertCompressed(row,col);
      else
        return insertUncompressed(row,col);
    }
    



    /** Must be called after inserting a set of non zero entries.
      */
    inline void finalize()
    {
      if(compressed())
      {
        Index size = static_cast<Index>(m_data.size());
        Index i = m_outerSize;
        // find the last filled column
        while (i>=0 && m_outerIndex[i]==0)
          --i;
        ++i;
        while (i<=m_outerSize)
        {
          m_outerIndex[i] = size;
          ++i;
        }
      }
    }
    
    /** Turns the matrix into the \em compressed format.
      */
    void makeCompressed()
    {
      if(compressed())
        return;
      
      Index oldStart = m_outerIndex[1];
      m_outerIndex[1] = m_innerNonZeros[0];
      for(Index j=1; j<m_outerSize; ++j)
      {
        Index nextOldStart = m_outerIndex[j+1];
        std::ptrdiff_t offset = oldStart - m_outerIndex[j];
        if(offset>0)
        {
          for(Index k=0; k<m_innerNonZeros[j]; ++k)
          {
            m_data.index(m_outerIndex[j]+k) = m_data.index(oldStart+k);
            m_data.value(m_outerIndex[j]+k) = m_data.value(oldStart+k);
          }
        }
        m_outerIndex[j+1] = m_outerIndex[j] + m_innerNonZeros[j];
        oldStart = nextOldStart;
      }
      delete[] m_innerNonZeros;
      m_innerNonZeros = 0;
      m_data.resize(m_outerIndex[m_outerSize]);
      m_data.squeeze();
    }

    /** Suppress all nonzeros which are \b much \b smaller \b than \a reference under the tolerence \a epsilon */
    void prune(Scalar reference, RealScalar epsilon = NumTraits<RealScalar>::dummy_precision())
    {
      prune(default_prunning_func(reference,epsilon));
    }
    
    /** Suppress all nonzeros which do not satisfy the predicate \a keep.
      * The functor type \a KeepFunc must implement the following function:
      * \code
      * bool operator() (const Index& row, const Index& col, const Scalar& value) const;
      * \endcode
      * \sa prune(Scalar,RealScalar)
      */
    template<typename KeepFunc>
    void prune(const KeepFunc& keep = KeepFunc())
    {
      Index k = 0;
      for(Index j=0; j<m_outerSize; ++j)
      {
        Index previousStart = m_outerIndex[j];
        m_outerIndex[j] = k;
        Index end = m_outerIndex[j+1];
        for(Index i=previousStart; i<end; ++i)
        {
          if(keep(IsRowMajor?j:m_data.index(i), IsRowMajor?m_data.index(i):j, m_data.value(i)))
          {
            m_data.value(k) = m_data.value(i);
            m_data.index(k) = m_data.index(i);
            ++k;
          }
        }
      }
      m_outerIndex[m_outerSize] = k;
      m_data.resize(k,0);
    }

    /** Resizes the matrix to a \a rows x \a cols matrix and initializes it to zero
      * \sa resizeNonZeros(Index), reserve(), setZero()
      */
    void resize(Index rows, Index cols)
    {
      const Index outerSize = IsRowMajor ? rows : cols;
      m_innerSize = IsRowMajor ? cols : rows;
      m_data.clear();
      if (m_outerSize != outerSize || m_outerSize==0)
      {
        delete[] m_outerIndex;
        m_outerIndex = new Index [outerSize+1];
        m_outerSize = outerSize;
      }
      if(m_innerNonZeros)
      {
        delete[] m_innerNonZeros;
        m_innerNonZeros = 0;
      }
      memset(m_outerIndex, 0, (m_outerSize+1)*sizeof(Index));
    }

    /** \deprecated
      * \internal
      * Low level API
      * Resize the nonzero vector to \a size 
      *  */
    void resizeNonZeros(Index size)
    {
      // TODO remove this function
      m_data.resize(size);
    }

    /** Default constructor yielding an empty \c 0 \c x \c 0 matrix */
    inline SparseMatrix()
      : m_outerSize(-1), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      resize(0, 0);
    }

    /** Constructs a \a rows \c x \a cols empty matrix */
    inline SparseMatrix(Index rows, Index cols)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      resize(rows, cols);
    }

    /** Constructs a sparse matrix from the sparse expression \a other */
    template<typename OtherDerived>
    inline SparseMatrix(const SparseMatrixBase<OtherDerived>& other)
      : m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      *this = other.derived();
    }

    /** Copy constructor */
    inline SparseMatrix(const SparseMatrix& other)
      : Base(), m_outerSize(0), m_innerSize(0), m_outerIndex(0), m_innerNonZeros(0)
    {
      *this = other.derived();
    }

    /** Swap the content of two sparse matrices of same type (optimization) */
    inline void swap(SparseMatrix& other)
    {
      //EIGEN_DBG_SPARSE(std::cout << "SparseMatrix:: swap\n");
      std::swap(m_outerIndex, other.m_outerIndex);
      std::swap(m_innerSize, other.m_innerSize);
      std::swap(m_outerSize, other.m_outerSize);
      std::swap(m_innerNonZeros, other.m_innerNonZeros);
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
        if(m_innerNonZeros)
        {
          delete[] m_innerNonZeros;
          m_innerNonZeros = 0;
        }
        if(other.compressed())
        {
          memcpy(m_outerIndex, other.m_outerIndex, (m_outerSize+1)*sizeof(Index));
          m_data = other.m_data;
        }
        else
        {
          Base::operator=(other);
        }
      }
      return *this;
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    template<typename Lhs, typename Rhs>
    inline SparseMatrix& operator=(const SparseSparseProduct<Lhs,Rhs>& product)
    { return Base::operator=(product); }
    
    template<typename OtherDerived>
    inline SparseMatrix& operator=(const ReturnByValue<OtherDerived>& other)
    { return Base::operator=(other.derived()); }
    
    template<typename OtherDerived>
    inline SparseMatrix& operator=(const EigenBase<OtherDerived>& other)
    { return Base::operator=(other.derived()); }
    #endif

    template<typename OtherDerived>
    EIGEN_DONT_INLINE SparseMatrix& operator=(const SparseMatrixBase<OtherDerived>& other)
    {
      const bool needToTranspose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
      if (needToTranspose)
      {
        // two passes algorithm:
        //  1 - compute the number of coeffs per dest inner vector
        //  2 - do the actual copy/eval
        // Since each coeff of the rhs has to be evaluated twice, let's evaluate it if needed
        typedef typename internal::nested<OtherDerived,2>::type OtherCopy;
        typedef typename internal::remove_all<OtherCopy>::type _OtherCopy;
        OtherCopy otherCopy(other.derived());

        resize(other.rows(), other.cols());
        Eigen::Map<Matrix<Index, Dynamic, 1> > (m_outerIndex,outerSize()).setZero();
        // pass 1
        // FIXME the above copy could be merged with that pass
        for (Index j=0; j<otherCopy.outerSize(); ++j)
          for (typename _OtherCopy::InnerIterator it(otherCopy, j); it; ++it)
            ++m_outerIndex[it.index()];

        // prefix sum
        Index count = 0;
        VectorXi positions(outerSize());
        for (Index j=0; j<outerSize(); ++j)
        {
          Index tmp = m_outerIndex[j];
          m_outerIndex[j] = count;
          positions[j] = count;
          count += tmp;
        }
        m_outerIndex[outerSize()] = count;
        // alloc
        m_data.resize(count);
        // pass 2
        for (Index j=0; j<otherCopy.outerSize(); ++j)
        {
          for (typename _OtherCopy::InnerIterator it(otherCopy, j); it; ++it)
          {
            Index pos = positions[it.index()]++;
            m_data.index(pos) = j;
            m_data.value(pos) = it.value();
          }
        }
        return *this;
      }
      else
      {
        // there is no special optimization
        return SparseMatrixBase<SparseMatrix>::operator=(other.derived());
      }
    }

    friend std::ostream & operator << (std::ostream & s, const SparseMatrix& m)
    {
      EIGEN_DBG_SPARSE(
        s << "Nonzero entries:\n";
        for (Index i=0; i<m.nonZeros(); ++i)
        {
          s << "(" << m.m_data.value(i) << "," << m.m_data.index(i) << ") ";
        }
        s << std::endl;
        s << std::endl;
        s << "Column pointers:\n";
        for (Index i=0; i<m.outerSize(); ++i)
        {
          s << m.m_outerIndex[i] << " ";
        }
        s << " $" << std::endl;
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

    /** Overloaded for performance */
    Scalar sum() const;
    
#   ifdef EIGEN_SPARSEMATRIX_PLUGIN
#     include EIGEN_SPARSEMATRIX_PLUGIN
#   endif

protected:
    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_DONT_INLINE Scalar& insertCompressed(Index row, Index col)
    {
      eigen_assert(compressed());

      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      Index previousOuter = outer;
      if (m_outerIndex[outer+1]==0)
      {
        // we start a new inner vector
        while (previousOuter>=0 && m_outerIndex[previousOuter]==0)
        {
          m_outerIndex[previousOuter] = static_cast<Index>(m_data.size());
          --previousOuter;
        }
        m_outerIndex[outer+1] = m_outerIndex[outer];
      }

      // here we have to handle the tricky case where the outerIndex array
      // starts with: [ 0 0 0 0 0 1 ...] and we are inserting in, e.g.,
      // the 2nd inner vector...
      bool isLastVec = (!(previousOuter==-1 && m_data.size()!=0))
                    && (size_t(m_outerIndex[outer+1]) == m_data.size());

      size_t startId = m_outerIndex[outer];
      // FIXME let's make sure sizeof(long int) == sizeof(size_t)
      size_t p = m_outerIndex[outer+1];
      ++m_outerIndex[outer+1];

      float reallocRatio = 1;
      if (m_data.allocatedSize()<=m_data.size())
      {
        // if there is no preallocated memory, let's reserve a minimum of 32 elements
        if (m_data.size()==0)
        {
          m_data.reserve(32);
        }
        else
        {
          // we need to reallocate the data, to reduce multiple reallocations
          // we use a smart resize algorithm based on the current filling ratio
          // in addition, we use float to avoid integers overflows
          float nnzEstimate = float(m_outerIndex[outer])*float(m_outerSize)/float(outer+1);
          reallocRatio = (nnzEstimate-float(m_data.size()))/float(m_data.size());
          // furthermore we bound the realloc ratio to:
          //   1) reduce multiple minor realloc when the matrix is almost filled
          //   2) avoid to allocate too much memory when the matrix is almost empty
          reallocRatio = (std::min)((std::max)(reallocRatio,1.5f),8.f);
        }
      }
      m_data.resize(m_data.size()+1,reallocRatio);

      if (!isLastVec)
      {
        if (previousOuter==-1)
        {
          // oops wrong guess.
          // let's correct the outer offsets
          for (Index k=0; k<=(outer+1); ++k)
            m_outerIndex[k] = 0;
          Index k=outer+1;
          while(m_outerIndex[k]==0)
            m_outerIndex[k++] = 1;
          while (k<=m_outerSize && m_outerIndex[k]!=0)
            m_outerIndex[k++]++;
          p = 0;
          --k;
          k = m_outerIndex[k]-1;
          while (k>0)
          {
            m_data.index(k) = m_data.index(k-1);
            m_data.value(k) = m_data.value(k-1);
            k--;
          }
        }
        else
        {
          // we are not inserting into the last inner vec
          // update outer indices:
          Index j = outer+2;
          while (j<=m_outerSize && m_outerIndex[j]!=0)
            m_outerIndex[j++]++;
          --j;
          // shift data of last vecs:
          Index k = m_outerIndex[j]-1;
          while (k>=Index(p))
          {
            m_data.index(k) = m_data.index(k-1);
            m_data.value(k) = m_data.value(k-1);
            k--;
          }
        }
      }

      while ( (p > startId) && (m_data.index(p-1) > inner) )
      {
        m_data.index(p) = m_data.index(p-1);
        m_data.value(p) = m_data.value(p-1);
        --p;
      }

      m_data.index(p) = inner;
      return (m_data.value(p) = 0);
    }

    class SingletonVector
    {
        Index m_index;
        Index m_value;
      public:
        typedef Index value_type;
        SingletonVector(Index i, Index v)
          : m_index(i), m_value(v)
        {}

        Index operator[](Index i) const { return i==m_index ? m_value : 0; }
    };

    /** \internal
      * \sa insert(Index,Index) */
    EIGEN_DONT_INLINE Scalar& insertUncompressed(Index row, Index col)
    {
      eigen_assert(!compressed());

      const Index outer = IsRowMajor ? row : col;
      const Index inner = IsRowMajor ? col : row;

      std::ptrdiff_t room = m_outerIndex[outer+1] - m_outerIndex[outer];
      std::ptrdiff_t innerNNZ = m_innerNonZeros[outer];
      if(innerNNZ>=room)
      {
        // this inner vector is full, we need to reallocate the whole buffer :(
        reserve(SingletonVector(outer,std::max<std::ptrdiff_t>(2,innerNNZ)));
      }

      Index startId = m_outerIndex[outer];
      Index p = startId + m_innerNonZeros[outer];
      while ( (p > startId) && (m_data.index(p-1) > inner) )
      {
        m_data.index(p) = m_data.index(p-1);
        m_data.value(p) = m_data.value(p-1);
        --p;
      }

      m_innerNonZeros[outer]++;

      m_data.index(p) = inner;
      return (m_data.value(p) = 0);
    }

private:
  struct default_prunning_func {
    default_prunning_func(Scalar ref, RealScalar eps) : reference(ref), epsilon(eps) {}
    inline bool operator() (const Index&, const Index&, const Scalar& value) const
    {
      return !internal::isMuchSmallerThan(value, reference, epsilon);
    }
    Scalar reference;
    RealScalar epsilon;
  };
};

template<typename Scalar, int _Options, typename _Index>
class SparseMatrix<Scalar,_Options,_Index>::InnerIterator
{
  public:
    InnerIterator(const SparseMatrix& mat, Index outer)
      : m_values(mat._valuePtr()), m_indices(mat._innerIndexPtr()), m_outer(outer), m_id(mat.m_outerIndex[outer]),
        m_end(mat.m_outerIndex[outer+1])
    {}

    inline InnerIterator& operator++() { m_id++; return *this; }

    inline const Scalar& value() const { return m_values[m_id]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_values[m_id]); }

    inline Index index() const { return m_indices[m_id]; }
    inline Index outer() const { return m_outer; }
    inline Index row() const { return IsRowMajor ? m_outer : index(); }
    inline Index col() const { return IsRowMajor ? index() : m_outer; }

    inline operator bool() const { return (m_id < m_end); }

  protected:
    const Scalar* m_values;
    const Index* m_indices;
    const Index m_outer;
    Index m_id;
    const Index m_end;
};

#endif // EIGEN_SPARSEMATRIX_H
