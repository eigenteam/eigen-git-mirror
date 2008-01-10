// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_NUMTRAITS_H
#define EIGEN_NUMTRAITS_H

/** \class NumTraits
  *
  * \brief Holds some data about the various numeric (i.e. scalar) types allowed by Eigen.
  *
  * \param T the numeric type about which this class provides data. Recall that Eigen allows
  *          only the following types for \a T: \c int, \c float, \c double,
  *          \c std::complex<float>, \c std::complex<double>.
  *
  * The provided data consists of:
  * \li A typedef \a Real, giving the "real part" type of \a T. If \a T is already real,
  *     then \a Real is just a typedef to \a T. If \a T is \c std::complex<U> then \a Real
  *     is a typedef to \a U.
  * \li A typedef \a FloatingPoint, giving the "floating-point type" of \a T. If \a T is
  *     \c int, then \a FloatingPoint is a typedef to \c double. Otherwise, \a FloatingPoint
  *     is a typedef to \a T.
  * \li An enum value \a IsComplex. It is equal to 1 if \a T is a \c std::complex
  *     type, and to 0 otherwise.
  * \li An enum \a HasFloatingPoint. It is equal to \c 0 if \a T is \c int,
  *     and to \c 1 otherwise.
  */
template<typename T> struct NumTraits;

template<> struct NumTraits<int>
{
  typedef int Real;
  typedef double FloatingPoint;
  enum {
    IsComplex = 0,
    HasFloatingPoint = 0
  };
};

template<> struct NumTraits<float>
{
  typedef float Real;
  typedef float FloatingPoint;
  enum {
    IsComplex = 0,
    HasFloatingPoint = 1
  };
};

template<> struct NumTraits<double>
{
  typedef double Real;
  typedef double FloatingPoint;
  enum {
    IsComplex = 0,
    HasFloatingPoint = 1
  };
};

template<typename _Real> struct NumTraits<std::complex<_Real> >
{
  typedef _Real Real;
  typedef std::complex<_Real> FloatingPoint;
  enum {
    IsComplex = 1,
    HasFloatingPoint = 1 // anyway we don't allow std::complex<int>
  };
};

#endif // EIGEN_NUMTRAITS_H
