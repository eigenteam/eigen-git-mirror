//=====================================================
// File   :  hand_vec_interface.hh
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
#ifndef HAND_VEC_INTERFACE_HH
#define HAND_VEC_INTERFACE_HH

#include <Eigen/Core>
#include "f77_interface.hh"

using namespace Eigen;

template<class real>
class hand_vec_interface : public f77_interface_base<real> {

public :

  typedef typename ei_packet_traits<real>::type Packet;
  static const int PacketSize = ei_packet_traits<real>::size;

  typedef typename f77_interface_base<real>::stl_matrix stl_matrix;
  typedef typename f77_interface_base<real>::stl_vector stl_vector;
  typedef typename f77_interface_base<real>::gene_matrix gene_matrix;
  typedef typename f77_interface_base<real>::gene_vector gene_vector;

  static void free_matrix(gene_matrix & A, int N){
    ei_aligned_free(A);
  }

  static void free_vector(gene_vector & B){
    ei_aligned_free(B);
  }

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    int N = A_stl.size();
    A = ei_aligned_malloc<real>(N*N);
    for (int j=0;j<N;j++)
      for (int i=0;i<N;i++)
        A[i+N*j] = A_stl[j][i];
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    int N = B_stl.size();
    B = ei_aligned_malloc<real>(N);
    for (int i=0;i<N;i++)
      B[i] = B_stl[i];
  }

  static inline std::string name() {
    #ifdef PEELING
    return "hand_vectorized_peeling";
    #else
    return "hand_vectorized";
    #endif
  }

  static inline void matrix_vector_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
  {
    int AN = (N/PacketSize)*PacketSize;
    for (int i=0;i<N;i++)
      X[i] = 0;
    for (int i=0;i<N;i++)
    {
      real tmp = B[i];
      Packet ptmp = ei_pset1(tmp);
      int iN = i*N;
      if (AN>0)
      {
        bool aligned = (iN % PacketSize) == 0;
        if (aligned)
        {
          #ifdef PEELING
          int ANP = (AN/(8*PacketSize))*8*PacketSize;
          for (int j = 0;j<ANP;j+=PacketSize*8)
          {
            ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_pload(&A[j+iN]))));
            ei_pstore(&X[j+PacketSize], ei_padd(ei_pload(&X[j+PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+PacketSize+iN]))));
            ei_pstore(&X[j+2*PacketSize], ei_padd(ei_pload(&X[j+2*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+2*PacketSize+iN]))));
            ei_pstore(&X[j+3*PacketSize], ei_padd(ei_pload(&X[j+3*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+3*PacketSize+iN]))));
            ei_pstore(&X[j+4*PacketSize], ei_padd(ei_pload(&X[j+4*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+4*PacketSize+iN]))));
            ei_pstore(&X[j+5*PacketSize], ei_padd(ei_pload(&X[j+5*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+5*PacketSize+iN]))));
            ei_pstore(&X[j+6*PacketSize], ei_padd(ei_pload(&X[j+6*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+6*PacketSize+iN]))));
            ei_pstore(&X[j+7*PacketSize], ei_padd(ei_pload(&X[j+7*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+7*PacketSize+iN]))));
          }
          for (int j = ANP;j<AN;j+=PacketSize)
            ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_pload(&A[j+iN]))));
          #else
          for (int j = 0;j<AN;j+=PacketSize)
            ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_pload(&A[j+iN]))));
          #endif
        }
        else
        {
          #ifdef PEELING
          int ANP = (AN/(8*PacketSize))*8*PacketSize;
          for (int j = 0;j<ANP;j+=PacketSize*8)
          {
            ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_ploadu(&A[j+iN]))));
            ei_pstore(&X[j+PacketSize], ei_padd(ei_pload(&X[j+PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+PacketSize+iN]))));
            ei_pstore(&X[j+2*PacketSize], ei_padd(ei_pload(&X[j+2*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+2*PacketSize+iN]))));
            ei_pstore(&X[j+3*PacketSize], ei_padd(ei_pload(&X[j+3*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+3*PacketSize+iN]))));
            ei_pstore(&X[j+4*PacketSize], ei_padd(ei_pload(&X[j+4*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+4*PacketSize+iN]))));
            ei_pstore(&X[j+5*PacketSize], ei_padd(ei_pload(&X[j+5*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+5*PacketSize+iN]))));
            ei_pstore(&X[j+6*PacketSize], ei_padd(ei_pload(&X[j+6*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+6*PacketSize+iN]))));
            ei_pstore(&X[j+7*PacketSize], ei_padd(ei_pload(&X[j+7*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+7*PacketSize+iN]))));
          }
          for (int j = ANP;j<AN;j+=PacketSize)
            ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_ploadu(&A[j+iN]))));
          #else
          for (int j = 0;j<AN;j+=PacketSize)
            ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_ploadu(&A[j+iN]))));
          #endif
        }
      }
      // process remaining scalars
      for (int j=AN;j<N;j++)
        X[j] += tmp * A[j+iN];
    }
  }

  static inline void atv_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
  {
    int AN = (N/PacketSize)*PacketSize;
    for (int i=0;i<N;i++)
      X[i] = 0;
    for (int i=0;i<N;i++)
    {
      real tmp = 0;
      Packet ptmp = ei_pset1(real(0));
      int iN = i*N;
      if (AN>0)
      {
        bool aligned = (iN % PacketSize) == 0;
        if (aligned)
        {
          #ifdef PEELING
          int ANP = (AN/(8*PacketSize))*8*PacketSize;
          for (int j = 0;j<ANP;j+=PacketSize*8)
          {
            ptmp =
              ei_padd(ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+PacketSize]), ei_pload(&A[j+PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+2*PacketSize]), ei_pload(&A[j+2*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+3*PacketSize]), ei_pload(&A[j+3*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+4*PacketSize]), ei_pload(&A[j+4*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+5*PacketSize]), ei_pload(&A[j+5*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+6*PacketSize]), ei_pload(&A[j+6*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+7*PacketSize]), ei_pload(&A[j+7*PacketSize+iN])),
              ptmp))))))));
          }
          for (int j = ANP;j<AN;j+=PacketSize)
            ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN])));
          #else
          for (int j = 0;j<AN;j+=PacketSize)
            ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN])));
          #endif
        }
        else
        {
          #ifdef PEELING
          int ANP = (AN/(8*PacketSize))*8*PacketSize;
          for (int j = 0;j<ANP;j+=PacketSize*8)
          {
            ptmp =
              ei_padd(ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+PacketSize]), ei_ploadu(&A[j+PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+2*PacketSize]), ei_ploadu(&A[j+2*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+3*PacketSize]), ei_ploadu(&A[j+3*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+4*PacketSize]), ei_ploadu(&A[j+4*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+5*PacketSize]), ei_ploadu(&A[j+5*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+6*PacketSize]), ei_ploadu(&A[j+6*PacketSize+iN])),
              ei_padd(ei_pmul(ei_pload(&B[j+7*PacketSize]), ei_ploadu(&A[j+7*PacketSize+iN])),
              ptmp))))))));
          }
          for (int j = ANP;j<AN;j+=PacketSize)
            ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN])));
          #else
          for (int j = 0;j<AN;j+=PacketSize)
            ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN])));
          #endif
        }
        tmp = ei_predux(ptmp);
      }
      // process remaining scalars
      for (int j=AN;j<N;j++)
        tmp += B[j] * A[j+iN];
      X[i] = tmp;
    }
  }

  static inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int N){
    int AN = (N/PacketSize)*PacketSize;
    if (AN>0)
    {
      Packet pcoef = ei_pset1(coef);
      #ifdef PEELING
      int ANP = (AN/(8*PacketSize))*8*PacketSize;
      for (int j = 0;j<ANP;j+=PacketSize*8)
      {
        ei_pstore(&Y[j             ], ei_padd(ei_pload(&Y[j             ]), ei_pmul(pcoef,ei_pload(&X[j             ]))));
        ei_pstore(&Y[j+  PacketSize], ei_padd(ei_pload(&Y[j+  PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+  PacketSize]))));
        ei_pstore(&Y[j+2*PacketSize], ei_padd(ei_pload(&Y[j+2*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+2*PacketSize]))));
        ei_pstore(&Y[j+3*PacketSize], ei_padd(ei_pload(&Y[j+3*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+3*PacketSize]))));
        ei_pstore(&Y[j+4*PacketSize], ei_padd(ei_pload(&Y[j+4*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+4*PacketSize]))));
        ei_pstore(&Y[j+5*PacketSize], ei_padd(ei_pload(&Y[j+5*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+5*PacketSize]))));
        ei_pstore(&Y[j+6*PacketSize], ei_padd(ei_pload(&Y[j+6*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+6*PacketSize]))));
        ei_pstore(&Y[j+7*PacketSize], ei_padd(ei_pload(&Y[j+7*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+7*PacketSize]))));
      }
      for (int j = ANP;j<AN;j+=PacketSize)
        ei_pstore(&Y[j], ei_padd(ei_pload(&Y[j]), ei_pmul(pcoef,ei_pload(&X[j]))));
      #else
      for (int j = 0;j<AN;j+=PacketSize)
        ei_pstore(&Y[j], ei_padd(ei_pload(&Y[j]), ei_pmul(pcoef,ei_pload(&X[j]))));
      #endif
    }
    // process remaining scalars
    for (int i=AN;i<N;i++)
      Y[i] += coef * X[i];
  }


};

#endif
