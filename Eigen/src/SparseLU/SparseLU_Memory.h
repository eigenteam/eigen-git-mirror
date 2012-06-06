// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
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

/* 
 
 * NOTE: This file is the modified version of [s,d,c,z]memory.c files in SuperLU 
 
 * -- SuperLU routine (version 3.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * August 1, 2008
 *
 * Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 *
 * THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
 * EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 *
 * Permission is hereby granted to use or copy this program for any
 * purpose, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is
 * granted, provided the above notices are retained, and a notice that
 * the code was modified is included with the above copyright notice.
 */

#ifndef EIGEN_SPARSELU_MEMORY
#define EIGEN_SPARSELU_MEMORY

#define LU_Reduce(alpha) ((alpha + 1) / 2) // i.e (alpha-1)/2 + 1
#define LU_GluIntArray(n) (5* (n) + 5)
#define LU_TempSpace(m, w) ( (2*w + 4 + LU_NO_MARKER) * m * sizeof(Index) \
                                  + (w + 1) * m * sizeof(Scalar)
namespace internal {
  
/**
 * \brief  Allocate various working space needed in the numerical factorization phase.
 * \param m number of rows of the input matrix 
 * \param n number of columns 
 * \param annz number of initial nonzeros in the matrix 
 * \param work scalar working space needed by all factor routines
 * \param iwork Integer working space 
 * \param lwork  if lwork=-1, this routine returns an estimated size of the required memory
 * \param Glu persistent data to facilitate multiple factors : will be deleted later ??
 * \return an estimated size of the required memory if lwork = -1; 
 *  FIXME should also return the size of actually allocated when memory allocation failed 
 * NOTE Unlike SuperLU, this routine does not allow the user to provide the size to allocate 
 */
template <typename ScalarVector,typename IndexVector>
int SparseLU::LUMemInit(int m, int n, int annz, Scalar *work, Index *iwork, int lwork, int fillratio, GlobalLU_t& Glu)
{
  typedef typename ScalarVector::Scalar; 
  typedef typename IndexVector::Index; 
  
  Glu.num_expansions = 0; //No memory expansions so far
  if (!Glu.expanders)
    Glu.expanders = new ExpHeader(LU_NBR_MEMTYPE); 
  
  // Guess the size for L\U factors 
  int nzlmax, nzumax, nzlumax; 
  nzumax = nzlumax = m_fillratio * annz; // estimated number of nonzeros in U 
  nzlmax  = std::max(1, m_fill_ratio/4.) * annz; // estimated  nnz in L factor

  // Return the estimated size to the user if necessary
  int estimated_size;
  if (lwork == IND_EMPTY) 
  {
    estimated_size = LU_GluIntArray(n) * sizeof(Index)  + LU_TempSpace(m, m_panel_size)
                    + (nzlmax + nzumax) * sizeof(Index) + (nzlumax+nzumax) *  sizeof(Scalar) + n); 
    return estimated_size;
  }
  
  // Setup the required space 
  // NOTE: In SuperLU, there is an option to let the user provide its own space, unlike here.
  
  // Allocate Integer pointers for L\U factors
  Glu.supno = new IndexVector; 
  Glu.supno->resize(n+1);
  
  Glu.xlsub = new IndexVector; 
  Glu.xlsub->resize(n+1);
  
  Glu.xlusup = new IndexVector; 
  Glu.xlusup->resize(n+1);
  
  Glu.xusub = new IndexVector; 
  Glu.xusub->resize(n+1);

  // Reserve memory for L/U factors
  Glu.lusup = new ScalarVector; 
  Glu.ucol = new ScalarVector; 
  Glu.lsub = new IndexVector;
  Glu.usub = new IndexVector; 
  
  expand<ScalarVector>(Glu.lusup,nzlumax, LUSUP, 0, 0, Glu); 
  expand<ScalarVector>(Glu.ucol,nzumax, UCOL, 0, 0, Glu); 
  expand<IndexVector>(Glu.lsub,nzlmax, LSUB, 0, 0, Glu); 
  expand<IndexVector>(Glu.usub,nzumax, USUB, 0, 1, Glu); 
  
  // Check if the memory is correctly allocated, 
  // Should be a try... catch section here 
  while ( !Glu.lusup.size() || !Glu.ucol.size() || !Glu.lsub.size() || !Glu.usub.size())
  {
    //otherwise reduce the estimated size and retry
//     delete [] Glu.lusup; 
//     delete [] Glu.ucol;
//     delete [] Glu.lsub;
//     delete [] Glu.usub;
//     
    nzlumax /= 2;
    nzumax /= 2;
    nzlmax /= 2;
    //FIXME Should be an excpetion here
    eigen_assert (nzlumax > annz && "Not enough memory to perform factorization");
    
    expand<ScalarVector>(Glu.lsup, nzlumax, LUSUP, 0, 0, Glu); 
    expand<ScalarVector>(Glu.ucol, nzumax, UCOL, 0, 0, Glu); 
    expand<IndexVector>(Glu.lsub, nzlmax, LSUB, 0, 0, Glu); 
    expand<IndexVector>(Glu.usub, nzumax, USUB, 0, 1, Glu); 
  }
  
  // LUWorkInit : Now, allocate known working storage
  int isize = (2 * m_panel_size + 3 + LU_NO_MARKER) * m + n;
  int dsize = m * m_panel_size + LU_NUM_TEMPV(m, m_panel_size, m_maxsuper, m_rowblk); 
  iwork = new Index(isize);
  eigen_assert( (m_iwork != 0) && "Malloc fails for iwork");
  work = new Scalar(dsize);
  eigen_assert( (m_work != 0) && "Malloc fails for dwork");
  
  ++Glu.num_expansions;
  return 0;
} // end LuMemInit

/** 
  * Expand the existing storage to accomodate more fill-ins
  * \param vec Valid pointer to a vector to allocate or expand
  * \param [in,out]prev_len At input, length from previous call. At output, length of the newly allocated vector
  * \param type Which part of the memory to expand
  * \param len_to_copy  Size of the memory to be copied to new store
  * \param keep_prev  true: use prev_len; Do not expand this vector; false: compute new_len and expand
  */
template <typename VectorType >
int  SparseLU::expand(VectorType& vec, int& prev_len, MemType type, int len_to_copy, bool keep_prev, GlobalLU_t& Glu) 
{
  
  float alpha = 1.5; // Ratio of the memory increase 
  int new_len; // New size of the allocated memory
  
  if(Glu.num_expansions == 0 || keep_prev) 
    new_len = prev_len; // First time allocate requested
  else 
    new_len = alpha * prev_len;
  
  // Allocate new space
//   vec = new VectorType(new_len); 
  VectorType old_vec(vec); 
  if ( Glu.num_expansions != 0 ) // The memory has been expanded before
  {
    int tries = 0; 
    vec.resize(new_len); //expand the current vector 
    if (keep_prev) 
    {
      if (!vec.size()) return -1 ; // FIXME could throw an exception somehow 
    }
    else 
    {
      while (!vec.size())
      {
        // Reduce the size and allocate again
        if ( ++tries > 10) return -1 
        alpha = LU_Reduce(alpha); 
        new_len = alpha * prev_len; 
        vec->resize(new_len); 
      }
    } // end allocation 
    //Copy the previous values to the newly allocated space 
    for (int i = 0; i < old_vec.size(); i++)
      vec(i) = old_vec(i); 
  } // end expansion 
//   expanders[type].mem = vec; 
//   expanders[type].size = new_len;
  prev_len = new_len;
  if(Glu.num_expansions) ++Glu.num_expansions;
  return 0; 
}

/** 
 * \brief Expand the existing storage 
 * 
 * NOTE: The calling sequence of this function is different from that of SuperLU
 * 
 * \return a pointer to the newly allocated space
 */
template <typename VectorType>
VectorType* SparseLU::LUMemXpand(int jcol, int next, MemType mem_type, int& maxlen)
{
  VectorType *newmem; 
  if (memtype == USUB)
    vec = expand<VectorType>(vec, maxlen, mem_type, next, 1);
  else
    vec = expand<VectorType>(vec, maxlen, mem_type, next, 0);
  // FIXME Should be an exception instead of an assert
  eigen_assert(new_mem.size() && "Can't expand memory"); 
  
  return new_mem;
  
}
    
}// Namespace Internal
#endif