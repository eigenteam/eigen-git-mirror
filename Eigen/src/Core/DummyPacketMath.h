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
template <typename Scalar> inline Scalar ei_padd(const Scalar&  a, const Scalar&  b) { return a + b; }

/** \internal \returns a - b (coeff-wise) */
template <typename Scalar> inline Scalar ei_psub(const Scalar&  a, const Scalar&  b) { return a - b; }

/** \internal \returns a * b (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmul(const Scalar&  a, const Scalar&  b) { return a * b; }

/** \internal \returns a / b (coeff-wise) */
template <typename Scalar> inline Scalar ei_pdiv(const Scalar&  a, const Scalar&  b) { return a / b; }

/** \internal \returns a * b - c (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmadd(const Scalar&  a, const Scalar&  b, const Scalar&  c)
{ return ei_padd(ei_pmul(a, b),c); }

/** \internal \returns the min of \a a and \a b  (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmin(const Scalar&  a, const Scalar&  b) { return std::min(a,b); }

/** \internal \returns the max of \a a and \a b  (coeff-wise) */
template <typename Scalar> inline Scalar ei_pmax(const Scalar&  a, const Scalar&  b) { return std::max(a,b); }

/** \internal \returns a packet version of \a *from, from must be 16 bytes aligned */
template <typename Scalar> inline Scalar ei_pload(const Scalar* from) { return *from; }

/** \internal \returns a packet version of \a *from, (un-aligned load) */
template <typename Scalar> inline Scalar ei_ploadu(const Scalar* from) { return *from; }

/** \internal \returns a packet with constant coefficients \a a, e.g.: (a,a,a,a) */
template <typename Scalar> inline Scalar ei_pset1(const Scalar& a) { return a; }

/** \internal copy the packet \a from to \a *to, \a to must be 16 bytes aligned */
template <typename Scalar> inline void ei_pstore(Scalar* to, const Scalar& from) { (*to) = from; }

/** \internal copy the packet \a from to \a *to, (un-aligned store) */
template <typename Scalar> inline void ei_pstoreu(Scalar* to, const Scalar& from) { (*to) = from; }

/** \internal \returns the first element of a packet */
template <typename Scalar> inline Scalar ei_pfirst(const Scalar& a) { return a; }

/** \internal \returns a packet where the element i contains the sum of the packet of \a vec[i] */
template <typename Scalar> inline Scalar ei_preduxp(const Scalar* vecs) { return vecs[0]; }

/** \internal \returns the sum of the elements of \a a*/
template <typename Scalar> inline Scalar ei_predux(const Scalar& a) { return a; }

#endif // EIGEN_DUMMY_PACKET_MATH_H

