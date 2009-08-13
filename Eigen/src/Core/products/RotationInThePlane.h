// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_ROTATION_IN_THE_PLANE_H
#define EIGEN_ROTATION_IN_THE_PLANE_H

/**********************************************************************
* This file implement ...
**********************************************************************/

template<typename Scalar, int Incr>
struct ei_apply_rotation_in_the_plane_selector;

template<typename VectorX, typename VectorY>
void ei_apply_rotation_in_the_plane(VectorX& x, VectorY& y, typename VectorX::Scalar c, typename VectorY::Scalar s)
{  
  ei_assert(x.size() == y.size());
  int size = x.size();
  int incrx = size ==1 ? 1 : &x.coeffRef(1) - &x.coeffRef(0);
  int incry = size ==1 ? 1 : &y.coeffRef(1) - &y.coeffRef(0);
  if (incrx==1 && incry==1)
    ei_apply_rotation_in_the_plane_selector<typename VectorX::Scalar,1>
      ::run(&x.coeffRef(0), &y.coeffRef(0), x.size(), c, s, 1, 1);
  else
    ei_apply_rotation_in_the_plane_selector<typename VectorX::Scalar,Dynamic>
      ::run(&x.coeffRef(0), &y.coeffRef(0), x.size(), c, s, incrx, incry);
}

template<typename Scalar>
struct ei_apply_rotation_in_the_plane_selector<Scalar,Dynamic>
{
  static void run(Scalar* x, Scalar* y, int size, Scalar c, Scalar s, int incrx, int incry)
  {
    for(int i=0; i<size; ++i)
    {
      Scalar xi = *x;
      Scalar yi = *y;
      *x = c * xi - s * yi;
      *y = s * xi + c * yi;
      x += incrx;
      y += incry;
    }
  }
};

// both vectors are sequentially stored in memory => vectorization
template<typename Scalar>
struct ei_apply_rotation_in_the_plane_selector<Scalar,1>
{
  static void run(Scalar* x, Scalar* y, int size, Scalar c, Scalar s, int, int)
  {
    typedef typename ei_packet_traits<Scalar>::type Packet;
    enum { PacketSize = ei_packet_traits<Scalar>::size, Peeling = 2 };
    int alignedStart = ei_alignmentOffset(y, size);
    int alignedEnd = alignedStart + ((size-alignedStart)/(Peeling*PacketSize))*(Peeling*PacketSize);

    const Packet pc = ei_pset1(c);
    const Packet ps = ei_pset1(s);
    
    for(int i=0; i<alignedStart; ++i)
    {
      Scalar xi = x[i];
      Scalar yi = y[i];
      x[i] = c * xi - s * yi;
      y[i] = s * xi + c * yi;
    }

    Scalar* px = x + alignedStart;
    Scalar* py = y + alignedStart;

    if(ei_alignmentOffset(x, size)==alignedStart)
      for(int i=alignedStart; i<alignedEnd; i+=PacketSize)
      {
        Packet xi = ei_pload(px);
        Packet yi = ei_pload(py);
        ei_pstore(px, ei_psub(ei_pmul(pc,xi),ei_pmul(ps,yi)));
        ei_pstore(py, ei_padd(ei_pmul(ps,xi),ei_pmul(pc,yi)));
        px += PacketSize;
        py += PacketSize;
      }
    else
      for(int i=alignedStart; i<alignedEnd; i+=Peeling*PacketSize)
      {
        Packet xi   = ei_ploadu(px);
        Packet xi1  = ei_ploadu(px+PacketSize);
        Packet yi   = ei_pload (py);
        Packet yi1  = ei_pload (py+PacketSize);
        ei_pstoreu(px, ei_psub(ei_pmul(pc,xi),ei_pmul(ps,yi)));
        ei_pstoreu(px+PacketSize, ei_psub(ei_pmul(pc,xi1),ei_pmul(ps,yi1)));
        ei_pstore (py, ei_padd(ei_pmul(ps,xi),ei_pmul(pc,yi)));
        ei_pstore (py+PacketSize, ei_padd(ei_pmul(ps,xi1),ei_pmul(pc,yi1)));
        px += Peeling*PacketSize;
        py += Peeling*PacketSize;
      }

    for(int i=alignedEnd; i<size; ++i)
    {
      Scalar xi = x[i];
      Scalar yi = y[i];
      x[i] = c * xi - s * yi;
      y[i] = s * xi + c * yi;
    }
  }
};

#endif // EIGEN_ROTATION_IN_THE_PLANE_H
