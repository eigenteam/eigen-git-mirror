//=====================================================
// File   :  ATLAS_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:21 CEST 2002
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
#ifndef ATLAS_PRODUIT_MATRICE_VECTEUR_HH
#define ATLAS_PRODUIT_MATRICE_VECTEUR_HH
#include "f77_interface_base.hh"
#include <string>
extern "C"
{
#include <atlas_level1.h>
#include <atlas_level2.h>
#include <atlas_level3.h>
#include "cblas.h"

}

template<class real>
class ATLAS_interface : public f77_interface_base<real>
{
public :

  typedef typename f77_interface_base<real>::gene_matrix gene_matrix;
  typedef typename f77_interface_base<real>::gene_vector gene_vector;

  static inline std::string name( void )
  {
    return "ATLAS";
  }

  static  inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    
    ATL_dgemv(CblasNoTrans,N,N,1.0,A,N,B,1,0.0,X,1);
    
  }

  static  inline void matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N)
  {    
    ATL_dgemm(CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,X,N);  
  }

  static  inline void ata_product(gene_matrix & A, gene_matrix & X, int N)
  {    
    ATL_dgemm(CblasTrans,CblasNoTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);  
  }

  static  inline void aat_product(gene_matrix & A, gene_matrix & X, int N)
  {    
    ATL_dgemm(CblasNoTrans,CblasTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);  
  }



  static  inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int N)  
  {
    ATL_daxpy(N,coef,X,1,Y,1);
  }
};

template<>
class ATLAS_interface<float> : public f77_interface_base<float>
{
public :

  static inline std::string name( void )
  {
    return "ATLAS";
  }

  static  inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    
    ATL_sgemv(CblasNoTrans,N,N,1.0,A,N,B,1,0.0,X,1);
    
  }

  static  inline void matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N)
  {
    ATL_sgemm(CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,X,N);
    
  }

  static  inline void ata_product(gene_matrix & A, gene_matrix & X, int N)
  {    
    ATL_sgemm(CblasTrans,CblasNoTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);  
  }

  static  inline void aat_product(gene_matrix & A, gene_matrix & X, int N)
  {    
    ATL_sgemm(CblasNoTrans,CblasTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);  
  }


  static inline void axpy(float coef, const gene_vector & X, gene_vector & Y, int N)  
  {
    ATL_saxpy(N,coef,X,1,Y,1);
  }
};


#endif



