// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_DUMMY_PACKET_MATH_H
#define EIGEN_DUMMY_PACKET_MATH_H

// Default implementation for types not supported by the vectorization.
// In practice these functions are provided to make easier the writting
// of generic vectorized code. However, at runtime, they should never be
// called, TODO so sould we raise an assertion or not ?
/** \internal \returns a + b (coeff-wise) */
template <typename Packet> inline Packet
ei_padd(const Packet&,
        const Packet&) { Packet ret; return ret; }

/** \internal \returns a - b (coeff-wise) */
template <typename Packet> inline Packet
ei_psub(const Packet&,
        const Packet&) { Packet ret; return ret; }

/** \internal \returns a * b (coeff-wise) */
template <typename Packet> inline Packet
ei_pmul(const Packet&,
        const Packet&) { Packet ret; return ret; }

/** \internal \returns a / b (coeff-wise) */
template <typename Packet> inline Packet
ei_pdiv(const Packet&,
        const Packet&) { Packet ret; return ret; }

/** \internal \returns the min of \a a and \a b  (coeff-wise) */
template <typename Packet> inline Packet
ei_pmin(const Packet&,
        const Packet&) { Packet ret; return ret; }

/** \internal \returns the max of \a a and \a b  (coeff-wise) */
template <typename Packet> inline Packet
ei_pmax(const Packet&,
        const Packet&) { Packet ret; return ret; }

/** \internal \returns a packet version of \a *from, from must be 16 bytes aligned */
template <typename Scalar> inline typename ei_packet_traits<Scalar>::type
ei_pload(const Scalar*) { typename ei_packet_traits<Scalar>::type ret; return ret; }

/** \internal \returns a packet version of \a *from, (un-aligned load) */
template <typename Scalar> inline typename ei_packet_traits<Scalar>::type
ei_ploadu(const Scalar*) { typename ei_packet_traits<Scalar>::type ret; return ret; }

/** \internal \returns a packet with constant coefficients \a a, e.g.: (a,a,a,a) */
template <typename Scalar> inline typename ei_packet_traits<Scalar>::type
ei_pset1(const Scalar&) { typename ei_packet_traits<Scalar>::type ret; return ret; }

/** \internal copy the packet \a from to \a *to, \a to must be 16 bytes aligned */
template <typename Scalar, typename Packet> inline void ei_pstore(Scalar*, const Packet&) {}

/** \internal copy the packet \a from to \a *to, (un-aligned store) */
template <typename Scalar, typename Packet> inline void ei_pstoreu(Scalar*, const Packet&) {}

/** \internal \returns the first element of a packet */
template <typename Packet> inline typename ei_unpacket_traits<Packet>::type ei_pfirst(const Packet&)
{ typename ei_unpacket_traits<Packet>::type ret; return ret; }

/** \internal \returns a packet where the element i contains the sum of the packet of \a vec[i] */
template <typename Packet> inline Packet
ei_preduxp(const Packet*) { Packet ret; return ret; }

/** \internal \returns the sum of the elements of \a a*/
template <typename Packet> inline typename ei_unpacket_traits<Packet>::type ei_predux(const Packet&)
{ typename ei_unpacket_traits<Packet>::type ret; return ret; }


////////////


/** \internal \returns a * b + c (coeff-wise) */
template <typename Packet> inline Packet
ei_pmadd(const Packet&  a,
         const Packet&  b,
         const Packet&  c)
{ return ei_padd(ei_pmul(a, b),c); }

/** \internal \returns a packet version of \a *from. If LoadMode equals Aligned, \a from must be 16 bytes aligned */
template <typename Scalar, int LoadMode> inline typename ei_packet_traits<Scalar>::type ei_ploadt(const Scalar* from)
{
  if(LoadMode == Aligned)
    return ei_pload(from);
  else
    return ei_ploadu(from);
}

/** \internal copy the packet \a from to \a *to. If StoreMode equals Aligned, \a to must be 16 bytes aligned */
template <typename Scalar, typename Packet, int LoadMode> inline void ei_pstoret(Scalar* to, const Packet& from)
{
  if(LoadMode == Aligned)
    ei_pstore(to, from);
  else
    ei_pstoreu(to, from);  
}

#endif // EIGEN_DUMMY_PACKET_MATH_H

