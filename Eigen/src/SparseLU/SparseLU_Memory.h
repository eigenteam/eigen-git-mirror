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
  
/* Allocate various working space needed in the numerical factorization phase.
 * m_work : space fot the output data structures (lwork is the size)
 * m_Glu: persistent data to facilitate multiple factors : is it really necessary ??
 * NOTE Unlike SuperLU, this routine does not allow the user to provide the size to allocate 
 * nor it return an estimated amount of space required.
 * 
 * Useful  variables :
 * -  m_fillratio : Ratio of fill expected 
 * - lwork = -1 : return an estimated size of the required memory
 *           = 0 : Estimate and allocate the memory
 */
template <typename Scalar,typename Index>
int SparseLU::LUMemInit(int lwork)
{
  int iword = sizeof(Index); 
  int dword = sizeof(Scalar);
  int n = m_Glu.n = m_mat.cols();
  int m = m_mat.rows();
  m_Glu.num_expansions = 0; // No memory expansions so far ??
  int estimated_size;
  
  
  if (!m_Glu.expanders)
    m_Glu.expanders = new ExpHeader(NO_MEMTYPE); 
  
  if (m_fact_t != SamePattern_SameRowPerm) // Create space for a new factorization
  { 
    // Guess the size for L\U factors 
    int annz = m_mat.nonZeros();
    int nzlmax, nzumax, nzlumax; 
    nzumax = nzlumax = m_fillratio * annz; // ???
    nzlmax  = std::max(1, m_fill_ratio/4.) * annz; //???
  
    // Return the estimated size to the user if necessary
    if (lwork = -1) 
    {
      estimated_size = LU_GluIntArray(n) * iword + LU_TempSpace(m, m_panel_size)
                      + (nzlmax + nzumax) * iword + (nzlumax+nzumax) * dword + n); 
      return estimated_size;
    }
    
    // Setup the required space 
    // NOTE: In SuperLU, there is an option to let the user provide its own space.
    
    // Allocate Integer pointers for L\U factors.resize(n+1);
    m_Glu.supno.resize(n+1);
    m_Glu.xlsub.resize(n+1);
    m_Glu.xlusup.resize(n+1);
    m_Glu.xusub.resize(n+1);
  
    // Reserve memory for L/U factors
    m_Glu.lusup = internal::expand<Scalar>(nzlumax, LUSUP, 0, 0, m_Glu); 
    m_Glu.ucol = internal::expand<Scalar>(nzumax, UCOL, 0, 0, m_Glu); 
    m_Glu.lsub = internal::expand<Index>(nzlmax, LSUB, 0, 0, m_Glu); 
    m_Glu.usub = internal::expand<Index>(nzumax, USUB, 0, 1, m_Glu); 
    
    // Check if the memory is correctly allocated, 
    while ( !m_Glu.lusup || !m_Glu.ucol || !m_Glu.lsub || !m_Glu.usub)
    {
      //otherwise reduce the estimated size and retry
      delete [] m_Glu.lusup; 
      delete [] m_Glu.ucol;
      delete [] m_Glu.lsub;
      delete [] m_Glu.usub;
      
      nzlumax /= 2;
      nzumax /= 2;
      nzlmax /= 2;
      eigen_assert (nzlumax > annz && "Not enough memory to perform factorization");
      
      m_Glu.lusup = internal::expand<Scalar>(nzlumax, LUSUP, 0, 0, m_Glu); 
      m_Glu.ucol = internal::expand<Scalar>(nzumax, UCOL, 0, 0, m_Glu); 
      m_Glu.lsub = internal::expand<Index>(nzlmax, LSUB, 0, 0, m_Glu); 
      m_Glu.usub = internal::expand<Index>(nzumax, USUB, 0, 1, m_Glu); 
    }
  }
  else  // m_fact == SamePattern_SameRowPerm;
  {
    if (lwork = -1) 
    {
      estimated_size = LU_GluIntArray(n) * iword + LU_TempSpace(m, m_panel_size)
                      + (Glu.nzlmax + Glu.nzumax) * iword + (Glu.nzlumax+Glu.nzumax) * dword + n); 
      return estimated_size;
    }
    //  Use existing space from previous factorization
    // Unlike in SuperLU, this should not  be necessary here since m_Glu is persistent as a member of the class
    m_Glu.xsup = m_Lstore.sup_to_col; 
    m_Glu.supno = m_Lstore.col_to_sup;
    m_Glu.xlsub = m_Lstore.rowind_colptr;
    m_Glu.xlusup = m_Lstore.nzval_colptr;
    xusub = m_Ustore.outerIndexPtr();
    
    m_Glu.expanders[LSUB].size = m_Glu.nzlmax; // Maximum value from previous factorization
    m_Glu.expanders[LUSUP].size = m_Glu.nzlumax;
    m_Glu.expanders[USUB].size = GLu.nzumax;
    m_Glu.expanders[UCOL].size = m_Glu.nzumax;
    m_Glu.lsub = GLu.expanders[LSUB].mem = m_Lstore.rowind;
    m_Glu.lusup = GLu.expanders[LUSUP].mem = m_Lstore.nzval;
    GLu.usub = m_Glu.expanders[USUB].mem = m_Ustore.InnerIndexPtr();
    m_Glu.ucol = m_Glu.expanders[UCOL].mem = m_Ustore.valuePtr();
  }
  
  // LUWorkInit : Now, allocate known working storage
  int isize = (2 * m_panel_size + 3 + LU_NO_MARKER) * m + n;
  int dsize = m * m_panel_size + LU_NUM_TEMPV(m, m_panel_size, m_maxsuper, m_rowblk); 
  m_iwork = new Index(isize);
  eigen_assert( (m_iwork != 0) && "Malloc fails for iwork");
  m_work = new Scalar(dsize);
  eigen_assert( (m_work != 0) && "Malloc fails for dwork");
  
  ++m_Glu.num_expansions;
  return 0;
} // end LuMemInit

/** 
  * Expand the existing storage to accomodate more fill-ins
  */
template <typename DestType >
DestType* SparseLU::expand(int& prev_len, // Length from previous call
        MemType type, // Which part of the memory to expand
        int len_to_copy, // Size of the memory to be copied to new store
        int keep_prev) // = 1: use prev_len; Do not expand this vector
                      // = 0: compute new_len to expand)
{
  
  float alpha = 1.5; // Ratio of the memory increase 
  int new_len; // New size of the allocated memory
  if(m_Glu.num_expansions == 0 || keep_prev) 
    new_len = prev_len;
  else 
    new_len = alpha * prev_len;
  
  // Allocate new space 
  DestType *new_mem, *old_mem; 
  new_mem = new DestType(new_len); 
  if ( m_Glu.num_expansions != 0 ) // The memory has been expanded before
  {
    int tries = 0; 
    if (keep_prev) 
    {
      if (!new_mem) return 0; 
    }
    else 
    {
      while ( !new_mem)
      {
        // Reduce the size and allocate again
        if ( ++tries > 10) return 0; 
        alpha = LU_Reduce(alpha); 
        new_len = alpha * prev_len; 
        new_mem = new DestType(new_len);
      }
    } // keep_prev
    //Copy the previous values to the newly allocated space 
    ExpHeader<DestType>* expanders = m_Glu.expanders;
    std::memcpy(new_mem, expanders[type].mem, len_to_copy); 
    delete [] expanders[type].mem; 
  }
  expanders[type].mem = new_mem; 
  expanders[type].size = new_len;
  prev_len = new_len;
  if(m_Glu.num_expansions) ++m_Glu.num_expansions;
  return expanders[type].mem; 
}

/** 
 * \brief Expand the existing storage 
 * 
 * NOTE: The calling sequence of this function is different from that of SuperLU
 * 
 * \return a pointer to the newly allocated space
 */
template <typename DestType>
DestType* SparseLU::LUMemXpand(int jcol, int next, MemType mem_type, int& maxlen)
{
  DestType *newmem; 
  if (memtype == USUB)
    new_mem = expand<DestType>(maxlen, mem_type, next, 1);
  else
    new_mem = expand<DestType>(maxlen, mem_type, next, 0);
  eigen_assert(new_mem && "Can't expand memory");
  
  return new_mem;
  
}
    
}// Namespace Internal
#endif