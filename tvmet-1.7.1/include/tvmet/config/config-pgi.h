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
 * $Id: config-pgi.h,v 1.6 2004/06/10 17:11:46 opetzold Exp $
 */

#ifndef TVMET_CONFIG_PGI_H
#define TVMET_CONFIG_PGI_H

#if defined(__PGI)


   // obviously does have pgCC 5.1 (trial) no long double on sqrt
#  if defined(TVMET_HAVE_LONG_DOUBLE)
#    undef TVMET_HAVE_LONG_DOUBLE
#  endif


#else // !defined(__PGI)

   // paranoia
#  warning "config header included without defined __PGI"

#endif

#endif // TVMET_CONFIG_PGI_H

// Local Variables:
// mode:C++
// End:
