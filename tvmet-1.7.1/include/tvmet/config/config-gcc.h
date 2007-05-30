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
 * $Id: config-gcc.h,v 1.6 2004/06/08 16:19:32 opetzold Exp $
 */

#ifndef TVMET_CONFIG_GCC_H
#define TVMET_CONFIG_GCC_H

#if defined(__GNUC__)

   // force inline
#  define TVMET_CXX_ALWAYS_INLINE __attribute__((always_inline))

#else // !defined(__GNUC__)

   // paranoia
#  warning "config header for gnuc included without defined __GNUC__"

#endif

#endif // TVMET_CONFIG_GCC_H

// Local Variables:
// mode:C++
// End:
