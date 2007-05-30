/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: config-icc.h,v 1.8 2004/06/10 16:36:55 opetzold Exp $
 */

#ifndef TVMET_CONFIG_ICC_H
#define TVMET_CONFIG_ICC_H

#if defined(__INTEL_COMPILER)

  /* isnan/isinf hack
   *
   * The problem is related intel's 8.0 macros isnan and isinf,
   * they are expanded in this version and they are not compileable
   * therefore. We use a small hack here - disabling. This is
   * not an real solution, nor forever.
   * For a list of all defined symbols use icpc -E -dM prog1.cpp
   * or read /opt/intel/compiler80/doc/c_ug/index.htm.
   */
#  if (__INTEL_COMPILER == 800) || (__INTEL_COMPILER > 800)
#    define TVMET_NO_IEEE_MATH_ISNAN
#    define TVMET_NO_IEEE_MATH_ISINF
#  endif


   /*
    * disable compiler warnings
    */
#  pragma warning(disable:981) // operands are evaluated in unspecified order


   /*
    * force inline using gcc's compatibility mode
    */
#  if (__INTEL_COMPILER == 800) || (__INTEL_COMPILER > 800)
#    define TVMET_CXX_ALWAYS_INLINE __attribute__((always_inline))
#  endif

#else // !defined(__INTEL_COMPILER)

   // paranoia
#  warning "config header included without defined __INTEL_COMPILER"

#endif

#endif // TVMET_CONFIG_ICC_H

// Local Variables:
// mode:C++
// End:
