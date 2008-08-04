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
    asm("#begin matrix_vector_product");
    int AN = (N/PacketSize)*PacketSize;
    int ANP = (AN/(2*PacketSize))*2*PacketSize;
    int bound = (N/4)*4;
    for (int i=0;i<N;i++)
      X[i] = 0;

    for (int i=0;i<bound;i+=4)
    {
      register real* __restrict__ A0 = A + i*N;
      register real* __restrict__ A1 = A + (i+1)*N;
      register real* __restrict__ A2 = A + (i+2)*N;
      register real* __restrict__ A3 = A + (i+3)*N;

      Packet ptmp0 = ei_pset1(B[i]);
      Packet ptmp1 = ei_pset1(B[i+1]);
      Packet ptmp2 = ei_pset1(B[i+2]);
      Packet ptmp3 = ei_pset1(B[i+3]);
//       register Packet ptmp0, ptmp1, ptmp2, ptmp3;
//       asm(
//
//           "movss     (%[B],%[j],4), %[ptmp0]  \n\t"
//           "shufps   $0,%[ptmp0],%[ptmp0] \n\t"
//           "movss    4(%[B],%[j],4), %[ptmp1]  \n\t"
//           "shufps   $0,%[ptmp1],%[ptmp1] \n\t"
//           "movss    8(%[B],%[j],4), %[ptmp2]  \n\t"
//           "shufps   $0,%[ptmp2],%[ptmp2] \n\t"
//           "movss   12(%[B],%[j],4), %[ptmp3]  \n\t"
//           "shufps   $0,%[ptmp3],%[ptmp3] \n\t"
//           : [ptmp0] "=x" (ptmp0),
//             [ptmp1] "=x" (ptmp1),
//             [ptmp2] "=x" (ptmp2),
//             [ptmp3] "=x" (ptmp3)
//           : [B] "r" (B),
//             [j] "r" (size_t(i))
//           : );

      if (AN>0)
      {
//         for (size_t j = 0;j<ANP;j+=8)
//         {
//           asm(
//
//           "movaps     (%[A0],%[j],4), %%xmm8  \n\t"
//           "movaps   16(%[A0],%[j],4), %%xmm12 \n\t"
//           "movups     (%[A3],%[j],4), %%xmm11 \n\t"
//           "movups   16(%[A3],%[j],4), %%xmm15 \n\t"
//           "movups     (%[A2],%[j],4), %%xmm10 \n\t"
//           "movups   16(%[A2],%[j],4), %%xmm14 \n\t"
//           "movups     (%[A1],%[j],4), %%xmm9  \n\t"
//           "movups   16(%[A1],%[j],4), %%xmm13 \n\t"
//
//           "mulps %[ptmp0], %%xmm8  \n\t"
//           "addps (%[res0],%[j],4), %%xmm8  \n\t"
//           "mulps %[ptmp3], %%xmm11 \n\t"
//           "addps %%xmm11, %%xmm8  \n\t"
//           "mulps %[ptmp2], %%xmm10 \n\t"
//           "addps %%xmm10, %%xmm8  \n\t"
//           "mulps %[ptmp1], %%xmm9  \n\t"
//           "addps %%xmm9, %%xmm8   \n\t"
//           "movaps %%xmm8, (%[res0],%[j],4)  \n\t"
//
//           "mulps %[ptmp0], %%xmm12 \n\t"
//           "addps 16(%[res0],%[j],4), %%xmm12  \n\t"
//           "mulps %[ptmp3], %%xmm15 \n\t"
//           "addps %%xmm15, %%xmm12  \n\t"
//           "mulps %[ptmp2], %%xmm14 \n\t"
//           "addps %%xmm14, %%xmm12  \n\t"
//           "mulps %[ptmp1], %%xmm13 \n\t"
//           "addps %%xmm13, %%xmm12  \n\t"
//           "movaps %%xmm12, 16(%[res0],%[j],4) \n\t"
//           :
//           : [res0] "r" (X), [j] "r" (j),[A0] "r" (A0),
//             [A1] "r" (A1),
//             [A2] "r" (A2),
//             [A3] "r" (A3),
//             [ptmp0] "x" (ptmp0),
//             [ptmp1] "x" (ptmp1),
//             [ptmp2] "x" (ptmp2),
//             [ptmp3] "x" (ptmp3)
//           : "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15", "%r14");
//         }
          register Packet A00;
          register Packet A01;
          register Packet A02;
          register Packet A03;
          register Packet A10;
          register Packet A11;
          register Packet A12;
          register Packet A13;
          for (int j = 0;j<ANP;j+=2*PacketSize)
          {
//             A00 = ei_pload(&A0[j]);
//             A01 = ei_ploadu(&A1[j]);
//             A02 = ei_ploadu(&A2[j]);
//             A03 = ei_ploadu(&A3[j]);
//             A10 = ei_pload(&A0[j+PacketSize]);
//             A11 = ei_ploadu(&A1[j+PacketSize]);
//             A12 = ei_ploadu(&A2[j+PacketSize]);
//             A13 = ei_ploadu(&A3[j+PacketSize]);
//
//             A00 = ei_pmul(ptmp0, A00);
//             A01 = ei_pmul(ptmp1, A01);
//             A02 = ei_pmul(ptmp2, A02);
//             A03 = ei_pmul(ptmp3, A03);
//             A10 = ei_pmul(ptmp0, A10);
//             A11 = ei_pmul(ptmp1, A11);
//             A12 = ei_pmul(ptmp2, A12);
//             A13 = ei_pmul(ptmp3, A13);
//
//             A00 = ei_padd(A00,A01);
//             A02 = ei_padd(A02,A03);
//             A00 = ei_padd(A00,ei_pload(&X[j]));
//             A00 = ei_padd(A00,A02);
//             ei_pstore(&X[j],A00);
//
//             A10 = ei_padd(A10,A11);
//             A12 = ei_padd(A12,A13);
//             A10 = ei_padd(A10,ei_pload(&X[j+PacketSize]));
//             A10 = ei_padd(A10,A12);
//             ei_pstore(&X[j+PacketSize],A10);

            ei_pstore(&X[j],
              ei_padd(ei_pload(&X[j]),
                ei_padd(
                  ei_padd(ei_pmul(ptmp0,ei_pload(&A0[j])),ei_pmul(ptmp1,ei_ploadu(&A1[j]))),
                  ei_padd(ei_pmul(ptmp2,ei_ploadu(&A2[j])),ei_pmul(ptmp3,ei_ploadu(&A3[j]))) )));

            ei_pstore(&X[j+PacketSize],
              ei_padd(ei_pload(&X[j+PacketSize]),
                ei_padd(
                  ei_padd(ei_pmul(ptmp0,ei_pload(&A0[j+PacketSize])),ei_pmul(ptmp1,ei_ploadu(&A1[j+PacketSize]))),
                  ei_padd(ei_pmul(ptmp2,ei_ploadu(&A2[j+PacketSize])),ei_pmul(ptmp3,ei_ploadu(&A3[j+PacketSize]))) )));
          }
          for (int j = ANP;j<AN;j+=PacketSize)
            ei_pstore(&X[j],
              ei_padd(ei_pload(&X[j]),
                ei_padd(
                  ei_padd(ei_pmul(ptmp0,ei_pload(&A0[j])),ei_pmul(ptmp1,ei_ploadu(&A1[j]))),
                  ei_padd(ei_pmul(ptmp2,ei_ploadu(&A2[j])),ei_pmul(ptmp3,ei_ploadu(&A3[j]))) )));
      }
      // process remaining scalars
      for (int j=AN;j<N;j++)
        X[j] += ei_pfirst(ptmp0) * A0[j] + ei_pfirst(ptmp1) * A1[j] + ei_pfirst(ptmp2) * A2[j] + ei_pfirst(ptmp3) * A3[j];
    }
    for (int i=bound;i<N;i++)
    {
      real tmp0 = B[i];
      Packet ptmp0 = ei_pset1(tmp0);
      int iN0 = i*N;
      if (AN>0)
      {
        bool aligned0 = (iN0 % PacketSize) == 0;
        if (aligned0)
          for (int j = 0;j<AN;j+=PacketSize)
            ei_pstore(&X[j], ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),ei_pload(&X[j])));
        else
          for (int j = 0;j<AN;j+=PacketSize)
            ei_pstore(&X[j], ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),ei_pload(&X[j])));
      }
      // process remaining scalars
      for (int j=AN;j<N;j++)
        X[j] += tmp0 * A[j+iN0];
    }
    asm("#end matrix_vector_product");
  }

//   static inline void matrix_vector_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
//   {
//     asm("#begin matrix_vector_product");
//     int AN = (N/PacketSize)*PacketSize;
//     int ANP = (AN/(2*PacketSize))*2*PacketSize;
//     int bound = (N/4)*4;
//     for (int i=0;i<N;i++)
//       X[i] = 0;
//
//     for (int i=0;i<bound;i+=4)
//     {
//       real tmp0 = B[i];
//       Packet ptmp0 = ei_pset1(tmp0);
//       real tmp1 = B[i+1];
//       Packet ptmp1 = ei_pset1(tmp1);
//       real tmp2 = B[i+2];
//       Packet ptmp2 = ei_pset1(tmp2);
//       real tmp3 = B[i+3];
//       Packet ptmp3 = ei_pset1(tmp3);
//       int iN0 = i*N;
//       int iN1 = (i+1)*N;
//       int iN2 = (i+2)*N;
//       int iN3 = (i+3)*N;
//       if (AN>0)
//       {
// //         int aligned0 = (iN0 % PacketSize);
//         int aligned1 = (iN1 % PacketSize);
//
//         if (aligned1==0)
//         {
//           for (int j = 0;j<AN;j+=PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pload(&X[j]),
//                 ei_padd(
//                   ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),ei_pmul(ptmp1,ei_pload(&A[j+iN1]))),
//                   ei_padd(ei_pmul(ptmp2,ei_pload(&A[j+iN2])),ei_pmul(ptmp3,ei_pload(&A[j+iN3]))) )));
//           }
//         }
//         else if (aligned1==2)
//         {
//           for (int j = 0;j<AN;j+=PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pload(&X[j]),
//                 ei_padd(
//                   ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),ei_pmul(ptmp1,ei_ploadu(&A[j+iN1]))),
//                   ei_padd(ei_pmul(ptmp2,ei_pload(&A[j+iN2])),ei_pmul(ptmp3,ei_ploadu(&A[j+iN3]))) )));
//           }
//         }
//         else
//         {
//           for (int j = 0;j<ANP;j+=2*PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pload(&X[j]),
//                 ei_padd(
//                   ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),ei_pmul(ptmp1,ei_ploadu(&A[j+iN1]))),
//                   ei_padd(ei_pmul(ptmp2,ei_ploadu(&A[j+iN2])),ei_pmul(ptmp3,ei_ploadu(&A[j+iN3]))) )));
//
//             ei_pstore(&X[j+PacketSize],
//               ei_padd(ei_pload(&X[j+PacketSize]),
//                 ei_padd(
//                   ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+PacketSize+iN0])),ei_pmul(ptmp1,ei_ploadu(&A[j+PacketSize+iN1]))),
//                   ei_padd(ei_pmul(ptmp2,ei_ploadu(&A[j+PacketSize+iN2])),ei_pmul(ptmp3,ei_ploadu(&A[j+PacketSize+iN3]))) )));
//
// //             ei_pstore(&X[j+2*PacketSize],
// //               ei_padd(ei_pload(&X[j+2*PacketSize]),
// //                 ei_padd(
// //                   ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+2*PacketSize+iN0])),ei_pmul(ptmp1,ei_ploadu(&A[j+2*PacketSize+iN1]))),
// //                   ei_padd(ei_pmul(ptmp2,ei_ploadu(&A[j+2*PacketSize+iN2])),ei_pmul(ptmp3,ei_ploadu(&A[j+2*PacketSize+iN3]))) )));
// //
// //             ei_pstore(&X[j+3*PacketSize],
// //               ei_padd(ei_pload(&X[j+3*PacketSize]),
// //                 ei_padd(
// //                   ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+3*PacketSize+iN0])),ei_pmul(ptmp1,ei_ploadu(&A[j+3*PacketSize+iN1]))),
// //                   ei_padd(ei_pmul(ptmp2,ei_ploadu(&A[j+3*PacketSize+iN2])),ei_pmul(ptmp3,ei_ploadu(&A[j+3*PacketSize+iN3]))) )));
//
//           }
//           for (int j = ANP;j<AN;j+=PacketSize)
//             ei_pstore(&X[j],
//               ei_padd(ei_pload(&X[j]),
//                 ei_padd(
//                   ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),ei_pmul(ptmp1,ei_ploadu(&A[j+iN1]))),
//                   ei_padd(ei_pmul(ptmp2,ei_ploadu(&A[j+iN2])),ei_pmul(ptmp3,ei_ploadu(&A[j+iN3]))) )));
//         }
//       }
//       // process remaining scalars
//       for (int j=AN;j<N;j++)
//         X[j] += tmp0 * A[j+iN0] + tmp1 * A[j+iN1] + tmp2 * A[j+iN2] + tmp3 * A[j+iN3];
//     }
//     for (int i=bound;i<N;i++)
//     {
//       real tmp0 = B[i];
//       Packet ptmp0 = ei_pset1(tmp0);
//       int iN0 = i*N;
//       if (AN>0)
//       {
//         bool aligned0 = (iN0 % PacketSize) == 0;
//         if (aligned0)
//           for (int j = 0;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),ei_pload(&X[j])));
//         else
//           for (int j = 0;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),ei_pload(&X[j])));
//       }
//       // process remaining scalars
//       for (int j=AN;j<N;j++)
//         X[j] += tmp0 * A[j+iN0];
//     }
//     asm("#end matrix_vector_product");
//   }

//   static inline void matrix_vector_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
//   {
//     asm("#begin matrix_vector_product");
//     int AN = (N/PacketSize)*PacketSize;
//     for (int i=0;i<N;i++)
//       X[i] = 0;
//
//     for (int i=0;i<N;i+=2)
//     {
//       real tmp0 = B[i];
//       Packet ptmp0 = ei_pset1(tmp0);
//       real tmp1 = B[i+1];
//       Packet ptmp1 = ei_pset1(tmp1);
//       int iN0 = i*N;
//       int iN1 = (i+1)*N;
//       if (AN>0)
//       {
//         bool aligned0 = (iN0 % PacketSize) == 0;
//         bool aligned1 = (iN1 % PacketSize) == 0;
//
//         if (aligned0 && aligned1)
//         {
//           for (int j = 0;j<AN;j+=PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_pload(&A[j+iN1])),ei_pload(&X[j]))));
//           }
//         }
//         else if (aligned0)
//         {
//           for (int j = 0;j<AN;j+=PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_ploadu(&A[j+iN1])),ei_pload(&X[j]))));
//           }
//         }
//         else if (aligned1)
//         {
//           for (int j = 0;j<AN;j+=PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_pload(&A[j+iN1])),ei_pload(&X[j]))));
//           }
//         }
//         else
//         {
//           int ANP = (AN/(4*PacketSize))*4*PacketSize;
//           for (int j = 0;j<ANP;j+=4*PacketSize)
//           {
//             ei_pstore(&X[j],
//               ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_ploadu(&A[j+iN1])),ei_pload(&X[j]))));
//
//             ei_pstore(&X[j+PacketSize],
//               ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+PacketSize+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_ploadu(&A[j+PacketSize+iN1])),ei_pload(&X[j+PacketSize]))));
//
//             ei_pstore(&X[j+2*PacketSize],
//               ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+2*PacketSize+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_ploadu(&A[j+2*PacketSize+iN1])),ei_pload(&X[j+2*PacketSize]))));
//
//             ei_pstore(&X[j+3*PacketSize],
//               ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+3*PacketSize+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_ploadu(&A[j+3*PacketSize+iN1])),ei_pload(&X[j+3*PacketSize]))));
//           }
//           for (int j = ANP;j<AN;j+=PacketSize)
//             ei_pstore(&X[j],
//               ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),
//               ei_padd(ei_pmul(ptmp1,ei_ploadu(&A[j+iN1])),ei_pload(&X[j]))));
//         }
//       }
//       // process remaining scalars
//       for (int j=AN;j<N;j++)
//         X[j] += tmp0 * A[j+iN0] + tmp1 * A[j+iN1];
//     }
//     int remaining = (N/2)*2;
//     for (int i=remaining;i<N;i++)
//     {
//       real tmp0 = B[i];
//       Packet ptmp0 = ei_pset1(tmp0);
//       int iN0 = i*N;
//       if (AN>0)
//       {
//         bool aligned0 = (iN0 % PacketSize) == 0;
//         if (aligned0)
//           for (int j = 0;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pmul(ptmp0,ei_pload(&A[j+iN0])),ei_pload(&X[j])));
//         else
//           for (int j = 0;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pmul(ptmp0,ei_ploadu(&A[j+iN0])),ei_pload(&X[j])));
//       }
//       // process remaining scalars
//       for (int j=AN;j<N;j++)
//         X[j] += tmp0 * A[j+iN0];
//     }
//     asm("#end matrix_vector_product");
//   }

//   static inline void matrix_vector_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
//   {
//     asm("#begin matrix_vector_product");
//     int AN = (N/PacketSize)*PacketSize;
//     for (int i=0;i<N;i++)
//       X[i] = 0;
//     for (int i=0;i<N;i++)
//     {
//       real tmp = B[i];
//       Packet ptmp = ei_pset1(tmp);
//       int iN = i*N;
//       if (AN>0)
//       {
//         bool aligned = (iN % PacketSize) == 0;
//         if (aligned)
//         {
//           #ifdef PEELING
//           Packet A0, A1, A2, X0, X1, X2;
//           int ANP = (AN/(8*PacketSize))*8*PacketSize;
//           for (int j = 0;j<ANP;j+=PacketSize*8)
//           {
//             A0 = ei_pload(&A[j+iN]);
//             X0 = ei_pload(&X[j]);
//             A1 = ei_pload(&A[j+PacketSize+iN]);
//             X1 = ei_pload(&X[j+PacketSize]);
//             A2 = ei_pload(&A[j+2*PacketSize+iN]);
//             X2 = ei_pload(&X[j+2*PacketSize]);
//             ei_pstore(&X[j], ei_padd(X0, ei_pmul(ptmp,A0)));
//             A0 = ei_pload(&A[j+3*PacketSize+iN]);
//             X0 = ei_pload(&X[j+3*PacketSize]);
//             ei_pstore(&X[j+PacketSize], ei_padd(ei_pload(&X1), ei_pmul(ptmp,A1)));
//             A1 = ei_pload(&A[j+4*PacketSize+iN]);
//             X1 = ei_pload(&X[j+4*PacketSize]);
//             ei_pstore(&X[j+2*PacketSize], ei_padd(ei_pload(&X2), ei_pmul(ptmp,A2)));
//             A2 = ei_pload(&A[j+5*PacketSize+iN]);
//             X2 = ei_pload(&X[j+5*PacketSize]);
//             ei_pstore(&X[j+3*PacketSize], ei_padd(ei_pload(&X0), ei_pmul(ptmp,A0)));
//             A0 = ei_pload(&A[j+6*PacketSize+iN]);
//             X0 = ei_pload(&X[j+6*PacketSize]);
//             ei_pstore(&X[j+4*PacketSize], ei_padd(ei_pload(&X1), ei_pmul(ptmp,A1)));
//             A1 = ei_pload(&A[j+7*PacketSize+iN]);
//             X1 = ei_pload(&X[j+7*PacketSize]);
//             ei_pstore(&X[j+5*PacketSize], ei_padd(ei_pload(&X2), ei_pmul(ptmp,A2)));
//             ei_pstore(&X[j+6*PacketSize], ei_padd(ei_pload(&X0), ei_pmul(ptmp,A0)));
//             ei_pstore(&X[j+7*PacketSize], ei_padd(ei_pload(&X1), ei_pmul(ptmp,A1)));
// //
// //             ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_pload(&A[j+iN]))));
// //             ei_pstore(&X[j+PacketSize], ei_padd(ei_pload(&X[j+PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+PacketSize+iN]))));
// //             ei_pstore(&X[j+2*PacketSize], ei_padd(ei_pload(&X[j+2*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+2*PacketSize+iN]))));
// //             ei_pstore(&X[j+3*PacketSize], ei_padd(ei_pload(&X[j+3*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+3*PacketSize+iN]))));
// //             ei_pstore(&X[j+4*PacketSize], ei_padd(ei_pload(&X[j+4*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+4*PacketSize+iN]))));
// //             ei_pstore(&X[j+5*PacketSize], ei_padd(ei_pload(&X[j+5*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+5*PacketSize+iN]))));
// //             ei_pstore(&X[j+6*PacketSize], ei_padd(ei_pload(&X[j+6*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+6*PacketSize+iN]))));
// //             ei_pstore(&X[j+7*PacketSize], ei_padd(ei_pload(&X[j+7*PacketSize]), ei_pmul(ptmp,ei_pload(&A[j+7*PacketSize+iN]))));
//           }
//           for (int j = ANP;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_pload(&A[j+iN]))));
//           #else
//           for (int j = 0;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_pload(&A[j+iN]))));
//           #endif
//         }
//         else
//         {
//           #ifdef PEELING
//           int ANP = (AN/(8*PacketSize))*8*PacketSize;
//           for (int j = 0;j<ANP;j+=PacketSize*8)
//           {
//             ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_ploadu(&A[j+iN]))));
//             ei_pstore(&X[j+PacketSize], ei_padd(ei_pload(&X[j+PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+PacketSize+iN]))));
//             ei_pstore(&X[j+2*PacketSize], ei_padd(ei_pload(&X[j+2*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+2*PacketSize+iN]))));
//             ei_pstore(&X[j+3*PacketSize], ei_padd(ei_pload(&X[j+3*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+3*PacketSize+iN]))));
//             ei_pstore(&X[j+4*PacketSize], ei_padd(ei_pload(&X[j+4*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+4*PacketSize+iN]))));
//             ei_pstore(&X[j+5*PacketSize], ei_padd(ei_pload(&X[j+5*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+5*PacketSize+iN]))));
//             ei_pstore(&X[j+6*PacketSize], ei_padd(ei_pload(&X[j+6*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+6*PacketSize+iN]))));
//             ei_pstore(&X[j+7*PacketSize], ei_padd(ei_pload(&X[j+7*PacketSize]), ei_pmul(ptmp,ei_ploadu(&A[j+7*PacketSize+iN]))));
//           }
//           for (int j = ANP;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_ploadu(&A[j+iN]))));
//           #else
//           for (int j = 0;j<AN;j+=PacketSize)
//             ei_pstore(&X[j], ei_padd(ei_pload(&X[j]), ei_pmul(ptmp,ei_ploadu(&A[j+iN]))));
//           #endif
//         }
//       }
//       // process remaining scalars
//       for (int j=AN;j<N;j++)
//         X[j] += tmp * A[j+iN];
//     }
//     asm("#end matrix_vector_product");
//   }

    static inline void atv_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
  {
    int AN = (N/PacketSize)*PacketSize;
    int bound = (N/4)*4;
    for (int i=0;i<bound;i+=4)
    {
      real tmp0 = 0;
      Packet ptmp0 = ei_pset1(real(0));
      real tmp1 = 0;
      Packet ptmp1 = ei_pset1(real(0));
      real tmp2 = 0;
      Packet ptmp2 = ei_pset1(real(0));
      real tmp3 = 0;
      Packet ptmp3 = ei_pset1(real(0));
      int iN0 = i*N;
      int iN1 = (i+1)*N;
      int iN2 = (i+2)*N;
      int iN3 = (i+3)*N;
      if (AN>0)
      {
        int align1 = (iN1 % PacketSize);
        if (align1==0)
        {
          for (int j = 0;j<AN;j+=PacketSize)
          {
            Packet b = ei_pload(&B[j]);
            ptmp0 = ei_padd(ptmp0, ei_pmul(b, ei_pload(&A[j+iN0])));
            ptmp1 = ei_padd(ptmp1, ei_pmul(b, ei_pload(&A[j+iN1])));
            ptmp2 = ei_padd(ptmp2, ei_pmul(b, ei_pload(&A[j+iN2])));
            ptmp3 = ei_padd(ptmp3, ei_pmul(b, ei_pload(&A[j+iN3])));
          }
        }
        else if (align1==2)
        {
          for (int j = 0;j<AN;j+=PacketSize)
          {
            Packet b = ei_pload(&B[j]);
            ptmp0 = ei_padd(ptmp0, ei_pmul(b, ei_pload(&A[j+iN0])));
            ptmp1 = ei_padd(ptmp1, ei_pmul(b, ei_ploadu(&A[j+iN1])));
            ptmp2 = ei_padd(ptmp2, ei_pmul(b, ei_pload(&A[j+iN2])));
            ptmp3 = ei_padd(ptmp3, ei_pmul(b, ei_ploadu(&A[j+iN3])));
          }
        }
        else
        {
          for (int j = 0;j<AN;j+=PacketSize)
          {
            Packet b = ei_pload(&B[j]);
            ptmp0 = ei_padd(ptmp0, ei_pmul(b, ei_pload(&A[j+iN0])));
            ptmp1 = ei_padd(ptmp1, ei_pmul(b, ei_ploadu(&A[j+iN1])));
            ptmp2 = ei_padd(ptmp2, ei_pmul(b, ei_ploadu(&A[j+iN2])));
            ptmp3 = ei_padd(ptmp3, ei_pmul(b, ei_ploadu(&A[j+iN3])));
          }
        }
        tmp0 = ei_predux(ptmp0);
        tmp1 = ei_predux(ptmp1);
        tmp2 = ei_predux(ptmp2);
        tmp3 = ei_predux(ptmp3);
      }
      // process remaining scalars
      for (int j=AN;j<N;j++)
      {
        tmp0 += B[j] * A[j+iN0];
        tmp1 += B[j] * A[j+iN1];
        tmp2 += B[j] * A[j+iN2];
        tmp3 += B[j] * A[j+iN3];
      }
      X[i+0] = tmp0;
      X[i+1] = tmp1;
      X[i+2] = tmp2;
      X[i+3] = tmp3;
    }

    for (int i=bound;i<N;i++)
    {
      real tmp0 = 0;
      Packet ptmp0 = ei_pset1(real(0));
      int iN0 = i*N;
      if (AN>0)
      {
        if (iN0 % PacketSize==0)
          for (int j = 0;j<AN;j+=PacketSize)
            ptmp0 = ei_padd(ptmp0, ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN0])));
        else
          for (int j = 0;j<AN;j+=PacketSize)
            ptmp0 = ei_padd(ptmp0, ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN0])));
        tmp0 = ei_predux(ptmp0);
      }
      // process remaining scalars
      for (int j=AN;j<N;j++)
        tmp0 += B[j] * A[j+iN0];
      X[i+0] = tmp0;
    }
  }

//   static inline void atv_product(const gene_matrix & A, const gene_vector & B, gene_vector & X, int N)
//   {
//     int AN = (N/PacketSize)*PacketSize;
//     for (int i=0;i<N;i++)
//       X[i] = 0;
//     for (int i=0;i<N;i++)
//     {
//       real tmp = 0;
//       Packet ptmp = ei_pset1(real(0));
//       int iN = i*N;
//       if (AN>0)
//       {
//         bool aligned = (iN % PacketSize) == 0;
//         if (aligned)
//         {
//           #ifdef PEELING
//           int ANP = (AN/(8*PacketSize))*8*PacketSize;
//           for (int j = 0;j<ANP;j+=PacketSize*8)
//           {
//             ptmp =
//               ei_padd(ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+PacketSize]), ei_pload(&A[j+PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+2*PacketSize]), ei_pload(&A[j+2*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+3*PacketSize]), ei_pload(&A[j+3*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+4*PacketSize]), ei_pload(&A[j+4*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+5*PacketSize]), ei_pload(&A[j+5*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+6*PacketSize]), ei_pload(&A[j+6*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+7*PacketSize]), ei_pload(&A[j+7*PacketSize+iN])),
//               ptmp))))))));
//           }
//           for (int j = ANP;j<AN;j+=PacketSize)
//             ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN])));
//           #else
//           for (int j = 0;j<AN;j+=PacketSize)
//             ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_pload(&A[j+iN])));
//           #endif
//         }
//         else
//         {
//           #ifdef PEELING
//           int ANP = (AN/(8*PacketSize))*8*PacketSize;
//           for (int j = 0;j<ANP;j+=PacketSize*8)
//           {
//             ptmp =
//               ei_padd(ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+PacketSize]), ei_ploadu(&A[j+PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+2*PacketSize]), ei_ploadu(&A[j+2*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+3*PacketSize]), ei_ploadu(&A[j+3*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+4*PacketSize]), ei_ploadu(&A[j+4*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+5*PacketSize]), ei_ploadu(&A[j+5*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+6*PacketSize]), ei_ploadu(&A[j+6*PacketSize+iN])),
//               ei_padd(ei_pmul(ei_pload(&B[j+7*PacketSize]), ei_ploadu(&A[j+7*PacketSize+iN])),
//               ptmp))))))));
//           }
//           for (int j = ANP;j<AN;j+=PacketSize)
//             ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN])));
//           #else
//           for (int j = 0;j<AN;j+=PacketSize)
//             ptmp = ei_padd(ptmp, ei_pmul(ei_pload(&B[j]), ei_ploadu(&A[j+iN])));
//           #endif
//         }
//         tmp = ei_predux(ptmp);
//       }
//       // process remaining scalars
//       for (int j=AN;j<N;j++)
//         tmp += B[j] * A[j+iN];
//       X[i] = tmp;
//     }
//   }

  static inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int N){
    int AN = (N/PacketSize)*PacketSize;
    if (AN>0)
    {
      Packet pcoef = ei_pset1(coef);
      #ifdef PEELING
      const int peelSize = 3;
      int ANP = (AN/(peelSize*PacketSize))*peelSize*PacketSize;
      float* X1 = X + PacketSize;
      float* Y1 = Y + PacketSize;
      float* X2 = X + 2*PacketSize;
      float* Y2 = Y + 2*PacketSize;
      Packet x0,x1,x2,y0,y1,y2;
      for (int j = 0;j<ANP;j+=PacketSize*peelSize)
      {
        x0 = ei_pload(X+j);
        x1 = ei_pload(X1+j);
        x2 = ei_pload(X2+j);

        y0 = ei_pload(Y+j);
        y1 = ei_pload(Y1+j);
        y2 = ei_pload(Y2+j);

        y0 = ei_pmadd(pcoef, x0, y0);
        y1 = ei_pmadd(pcoef, x1, y1);
        y2 = ei_pmadd(pcoef, x2, y2);

        ei_pstore(Y+j,  y0);
        ei_pstore(Y1+j, y1);
        ei_pstore(Y2+j, y2);
//         ei_pstore(&Y[j+2*PacketSize], ei_padd(ei_pload(&Y[j+2*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+2*PacketSize]))));
//         ei_pstore(&Y[j+3*PacketSize], ei_padd(ei_pload(&Y[j+3*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+3*PacketSize]))));
//         ei_pstore(&Y[j+4*PacketSize], ei_padd(ei_pload(&Y[j+4*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+4*PacketSize]))));
//         ei_pstore(&Y[j+5*PacketSize], ei_padd(ei_pload(&Y[j+5*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+5*PacketSize]))));
//         ei_pstore(&Y[j+6*PacketSize], ei_padd(ei_pload(&Y[j+6*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+6*PacketSize]))));
//         ei_pstore(&Y[j+7*PacketSize], ei_padd(ei_pload(&Y[j+7*PacketSize]), ei_pmul(pcoef,ei_pload(&X[j+7*PacketSize]))));
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
