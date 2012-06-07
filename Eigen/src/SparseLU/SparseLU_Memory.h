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
 * \brief  Allocate various working space failed in the numerical factorization phase.
 * \param m number of rows of the input matrix 
 * \param n number of columns 
 * \param annz number of initial nonzeros in the matrix 
 * \param work scalar working space needed by all factor routines
 * \param iwork Integer working space 
 * \param lwork  if lwork=-1, this routine returns an estimated size of the required memory
 * \param Glu persistent data to facilitate multiple factors : will be deleted later ??
 * \return an estimated size of the required memory if lwork = -1; otherwise, return the size of actually allocated when memory allocation failed 
 * NOTE Unlike SuperLU, this routine does not allow the user to provide its own user space
 */
template <typename ScalarVector,typename IndexVector>
int SparseLU::LUMemInit(int m, int n, int annz, ScalarVector& work, IndexVector& iwork, int lwork, int fillratio, GlobalLU_t& Glu)
{
  typedef typename ScalarVector::Scalar; 
  typedef typename IndexVector::Index; 
  
  int& num_expansions = Glu.num_expansions; //No memory expansions so far
  num_expansions = 0; 
  // Guess the size for L\U factors 
  Index& nzlmax = Glu.nzlmax; 
  Index& nzumax = Glu.nzumax; 
  Index& nzlumax = Glu.nzlumax;
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
  
  // First allocate Integer pointers for L\U factors
  Glu.supno.resize(n+1);
  Glu.xlsub.resize(n+1);
  Glu.xlusup.resize(n+1);
  Glu.xusub.resize(n+1);

  // Reserve memory for L/U factors
  expand<ScalarVector>(Glu.lusup, nzlumax, 0, 0, num_expansions); 
  expand<ScalarVector>(Glu.ucol,nzumax, 0, 0, num_expansions); 
  expand<IndexVector>(Glu.lsub,nzlmax, 0, 0, num_expansions); 
  expand<IndexVector>(Glu.usub,nzumax, 0, 1, num_expansions);
  
  // Check if the memory is correctly allocated, 
  // Should be a try... catch section here 
  while ( !Glu.lusup.size() || !Glu.ucol.size() || !Glu.lsub.size() || !Glu.usub.size())
  {
    //otherwise reduce the estimated size and retry
    nzlumax /= 2;
    nzumax /= 2;
    nzlmax /= 2;
    //FIXME Should be an exception here
    if (nzlumax < annz ) return nzlumax; 
    
    expand<ScalarVector>(Glu.lsup, nzlumax, 0, 0, Glu); 
    expand<ScalarVector>(Glu.ucol, nzumax, 0, 0, Glu); 
    expand<IndexVector>(Glu.lsub, nzlmax, 0, 0, Glu); 
    expand<IndexVector>(Glu.usub, nzumax, 0, 1, Glu); 
  }
  
  // LUWorkInit : Now, allocate known working storage
  int isize = (2 * m_panel_size + 3 + LU_NO_MARKER) * m + n;
  int dsize = m * m_panel_size + LU_NUM_TEMPV(m, m_panel_size, m_maxsuper, m_rowblk); 
  iwork.resize(isize); 
  work.resize(isize); 
  
  ++num_expansions;
  return 0;
  
} // end LuMemInit

/** 
  * Expand the existing storage to accomodate more fill-ins
  * \param vec Valid pointer to the vector to allocate or expand
  * \param [in,out]length  At input, contain the current length of the vector that is to be increased. At output, length of the newly allocated vector
  * \param [in]len_to_copy Current number of elements in the factors
  * \param keep_prev  true: use length  and do not expand the vector; false: compute new_len and expand
  * \param [in,out]num_expansions Number of times the memory has been expanded
  */
template <typename VectorType >
int  SparseLU::expand(VectorType& vec, int& length, int len_to_copy, bool keep_prev, int& num_expansions) 
{
  
  float alpha = 1.5; // Ratio of the memory increase 
  int new_len; // New size of the allocated memory
  
  if(num_expansions == 0 || keep_prev) 
    new_len = length ; // First time allocate requested
  else 
    new_len = alpha * length ;
  
  VectorType old_vec; // Temporary vector to hold the previous values   
  if (len_to_copy > 0 )
    old_vec = vec; // old_vec should be of size len_to_copy... to be checked
  
  //expand the current vector //FIXME Should be in a try ... catch region
  vec.resize(new_len); 
  /* 
   * Test if the memory has been well allocated  
   * otherwise reduce the size and try to reallocate
   * copy data from previous vector (if exists) to the newly allocated vector 
   */
  if ( num_expansions != 0 ) // The memory has been expanded before
  {
    int tries = 0; 
    if (keep_prev) 
    {
      if (!vec.size()) return new_len ;
    }
    else 
    {
      while (!vec.size())
      {
       // Reduce the size and allocate again
        if ( ++tries > 10) return new_len;  
        alpha = LU_Reduce(alpha); 
        new_len = alpha * length ; 
        vec.resize(new_len); //FIXME Should be in a try catch section
      }
    } // end allocation 
    
    //Copy the previous values to the newly allocated space 
    if (len_to_copy > 0)
      vec.segment(0, len_to_copy) = old_vec;   
  } // end expansion 
  length  = new_len;
  if(num_expansions) ++num_expansions;
  return 0; 
}

/** 
 * \brief Expand the existing storage 
 * \param vec vector to expand 
 * \param [in,out]maxlen On input, previous size of vec (Number of elements to copy ). on output, new size
 * \param next current number of elements in the vector.
 * \param Glu Global data structure 
 * \return 0 on success, > 0 size of the memory allocated so far
 */
template <typename IndexVector>
int SparseLU::LUMemXpand(VectorType& vec, int& maxlen, int next, LU_MemType memtype, LU_GlobalLu_t& Glu)
{
  int failed_size; 
  int& num_expansions = Glu.num_expansions; 
  if (memtype == USUB)
     failed_size = expand<IndexVector>(vec, maxlen, next, 1, num_expansions);
  else
    failed_size = expand<IndexVector>(vec, maxlen, next, 0, num_expansions);

  if (failed_size)
    return faileld_size; 
  
  // The following code  is not really needed since maxlen is passed by reference 
  // and correspond to the appropriate field in Glu
//   switch ( mem_type ) {
//     case LUSUP:
//       Glu.nzlumax = maxlen;
//       break; 
//     case UCOL:
//       Glu.nzumax = maxlen; 
//       break; 
//     case LSUB:
//       Glu.nzlmax = maxlen; 
//       break;
//     case USUB:
//       Glu.nzumax = maxlen; 
//       break; 
//   }
  
  return 0 ; 
  
}
    
}// Namespace Internal
#endif