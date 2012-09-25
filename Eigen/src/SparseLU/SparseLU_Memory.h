// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

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

#define LU_NO_MARKER 3
#define LU_NUM_TEMPV(m,w,t,b) ((std::max)(m, (t+b)*w)  )
#define IND_EMPTY (-1)

#define LU_Reduce(alpha) ((alpha + 1) / 2) // i.e (alpha-1)/2 + 1
#define LU_GluIntArray(n) (5* (n) + 5)
#define LU_TempSpace(m, w) ( (2*w + 4 + LU_NO_MARKER) * m * sizeof(Index) \
                                  + (w + 1) * m * sizeof(Scalar) )


/** 
  * Expand the existing storage to accomodate more fill-ins
  * \param vec Valid pointer to the vector to allocate or expand
  * \param [in,out]length  At input, contain the current length of the vector that is to be increased. At output, length of the newly allocated vector
  * \param [in]nbElts Current number of elements in the factors
  * \param keep_prev  1: use length  and do not expand the vector; 0: compute new_len and expand
  * \param [in,out]num_expansions Number of times the memory has been expanded
  */
template <typename Scalar, typename Index>
template <typename VectorType>
int  SparseLUBase<Scalar,Index>::expand(VectorType& vec, int& length, int nbElts, int keep_prev, int& num_expansions) 
{
  
  float alpha = 1.5; // Ratio of the memory increase 
  int new_len; // New size of the allocated memory
  
  if(num_expansions == 0 || keep_prev) 
    new_len = length ; // First time allocate requested
  else 
    new_len = alpha * length ;
  
  VectorType old_vec; // Temporary vector to hold the previous values   
  if (nbElts > 0 )
    old_vec = vec.segment(0,nbElts); 
  
  //Allocate or expand the current vector
  try 
  {
    vec.resize(new_len); 
  }
  catch(std::bad_alloc& )
  {
    if ( !num_expansions )
    {
      // First time to allocate from LUMemInit()
      throw; // Pass the exception to LUMemInit() which has a try... catch block
    }
    if (keep_prev)
    {
      // In this case, the memory length should not not be reduced
      return new_len;
    }
    else 
    {
      // Reduce the size and increase again 
      int tries = 0; // Number of attempts
      do 
      {
        alpha = LU_Reduce(alpha);
        new_len = alpha * length ; 
        try
        {
          vec.resize(new_len); 
        }
        catch(std::bad_alloc& )
        {
          tries += 1; 
          if ( tries > 10) return new_len; 
        }
      } while (!vec.size());
    }
  }
  //Copy the previous values to the newly allocated space 
  if (nbElts > 0)
    vec.segment(0, nbElts) = old_vec;   
   
  
  length  = new_len;
  if(num_expansions) ++num_expansions;
  return 0; 
}

/**
 * \brief  Allocate various working space for the numerical factorization phase.
 * \param m number of rows of the input matrix 
 * \param n number of columns 
 * \param annz number of initial nonzeros in the matrix 
 * \param lwork  if lwork=-1, this routine returns an estimated size of the required memory
 * \param glu persistent data to facilitate multiple factors : will be deleted later ??
 * \return an estimated size of the required memory if lwork = -1; otherwise, return the size of actually allocated memory when allocation failed, and 0 on success
 * NOTE Unlike SuperLU, this routine does not support successive factorization with the same pattern and the same row permutation
 */
template <typename Scalar, typename Index>
int SparseLUBase<Scalar,Index>::LUMemInit(int m, int n, int annz, int lwork, int fillratio, int panel_size,  GlobalLU_t& glu)
{
  int& num_expansions = glu.num_expansions; //No memory expansions so far
  num_expansions = 0; 
  glu.nzumax = glu.nzlumax = std::max(fillratio * annz, m*n); // estimated number of nonzeros in U 
  glu.nzlmax  = std::max(1., fillratio/4.) * annz; // estimated  nnz in L factor

  // Return the estimated size to the user if necessary
  if (lwork == IND_EMPTY) 
  {
    int estimated_size;
    estimated_size = LU_GluIntArray(n) * sizeof(Index)  + LU_TempSpace(m, panel_size)
                    + (glu.nzlmax + glu.nzumax) * sizeof(Index) + (glu.nzlumax+glu.nzumax) *  sizeof(Scalar) + n; 
    return estimated_size;
  }
  
  // Setup the required space 
  
  // First allocate Integer pointers for L\U factors
  glu.xsup.resize(n+1);
  glu.supno.resize(n+1);
  glu.xlsub.resize(n+1);
  glu.xlusup.resize(n+1);
  glu.xusub.resize(n+1);

  // Reserve memory for L/U factors
  do 
  {
    try
    {
      expand<ScalarVector>(glu.lusup, glu.nzlumax, 0, 0, num_expansions); 
      expand<ScalarVector>(glu.ucol,glu.nzumax, 0, 0, num_expansions); 
      expand<IndexVector>(glu.lsub,glu.nzlmax, 0, 0, num_expansions); 
      expand<IndexVector>(glu.usub,glu.nzumax, 0, 1, num_expansions); 
    }
    catch(std::bad_alloc& )
    {
      //Reduce the estimated size and retry
      glu.nzlumax /= 2;
      glu.nzumax /= 2;
      glu.nzlmax /= 2;
      if (glu.nzlumax < annz ) return glu.nzlumax; 
    }
    
  } while (!glu.lusup.size() || !glu.ucol.size() || !glu.lsub.size() || !glu.usub.size());

  
  
  ++num_expansions;
  return 0;
  
} // end LuMemInit

/** 
 * \brief Expand the existing storage 
 * \param vec vector to expand 
 * \param [in,out]maxlen On input, previous size of vec (Number of elements to copy ). on output, new size
 * \param nbElts current number of elements in the vector.
 * \param glu Global data structure 
 * \return 0 on success, > 0 size of the memory allocated so far
 */
template <typename Scalar, typename Index>
template <typename VectorType>
int SparseLUBase<Scalar,Index>::LUMemXpand(VectorType& vec, int& maxlen, int nbElts, LU_MemType memtype, int& num_expansions)
{
  int failed_size; 
  if (memtype == USUB)
     failed_size = expand<VectorType>(vec, maxlen, nbElts, 1, num_expansions);
  else
    failed_size = expand<VectorType>(vec, maxlen, nbElts, 0, num_expansions);

  if (failed_size)
    return failed_size; 
  
  return 0 ; 
  
}
#endif