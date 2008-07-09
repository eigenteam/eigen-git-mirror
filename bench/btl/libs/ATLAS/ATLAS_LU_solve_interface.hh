//=====================================================
// File   :  ATLAS_LU_solve_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:22 CEST 2002
//=====================================================
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
// 
#ifndef ATLAS_LU_solve_interface_HH
#define ATLAS_LU_solve_interface_HH
#include "ATLAS_interface.hh"
extern "C"
{
#include <atlas_level1.h>
#include <atlas_level2.h>
#include <atlas_level3.h>
#include "cblas.h"
#include <atlas_lapack.h>

}

template<class real>
class ATLAS_LU_solve_interface : public ATLAS_interface<real>
{
public :

  typedef typename ATLAS_interface<real>::gene_matrix gene_matrix;
  typedef typename ATLAS_interface<real>::gene_vector gene_vector;

  typedef int *  Pivot_Vector;
  
  inline static void new_Pivot_Vector(Pivot_Vector & pivot, int N)
  {
    
    pivot = new int[N];

  }

  inline static void free_Pivot_Vector(Pivot_Vector & pivot)
  {
    
    delete pivot;

  }
  

  inline static void LU_factor(gene_matrix & LU, Pivot_Vector & pivot, int N)
  {
    
    int error=ATL_dgetrf(CblasColMajor,N,N,LU,N,pivot);    

  }

  inline static void  LU_solve(const gene_matrix & LU, const Pivot_Vector pivot, const gene_vector &B, gene_vector X, int N)
  {
        
    copy_vector(B,X,N);
    ATL_dgetrs(CblasColMajor,CblasNoTrans,N,1,LU,N,pivot,X,N);
    
  }

};

template<>
class ATLAS_LU_solve_interface<float> : public ATLAS_interface<float>
{
public :

  typedef int *  Pivot_Vector;
  
  inline static void new_Pivot_Vector(Pivot_Vector & pivot, int N)
  {
    
    pivot = new int[N];

  }

  inline static void free_Pivot_Vector(Pivot_Vector & pivot)
  {
    
    delete pivot;

  }
  

  inline static void LU_factor(gene_matrix & LU, Pivot_Vector & pivot, int N)
  {
    
    int error=ATL_sgetrf(CblasColMajor,N,N,LU,N,pivot);    

  }

  inline static void  LU_solve(const gene_matrix & LU, const Pivot_Vector pivot, const gene_vector &B, gene_vector X, int N)
  {
        
    copy_vector(B,X,N);
    ATL_sgetrs(CblasColMajor,CblasNoTrans,N,1,LU,N,pivot,X,N);
    
  }

};


#endif



