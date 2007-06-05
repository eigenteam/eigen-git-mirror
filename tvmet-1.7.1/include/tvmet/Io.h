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
 * $Id: Io.h,v 1.3 2004/04/30 16:03:38 opetzold Exp $
 */

#ifndef TVMET_IO_H
#define TVMET_IO_H

namespace tvmet {

/**
 * \class IoPrintHelper Io.h "tvmet/Io.h"
 * \brief Determines the number of digits regarding the sign of the
 *        container.
 *        This class is nesessary due to the complex type and the
 *        function min(), which are not defined for this type.
 *        So we have to dispatch between pod and complex types
 *        to get an information about the extra space for signs.
 */
template<class C>
class IoPrintHelper {
  IoPrintHelper();
  IoPrintHelper(const IoPrintHelper&);
  IoPrintHelper& operator=(const IoPrintHelper&);

private:
  static std::streamsize width(const C& e) {
    std::streamsize w = static_cast<std::streamsize>(10); //FIXME arbitrary value
    return w > 0 ? w : 0;
  }

public:
  static std::streamsize width(dispatch<true>, const C& e) {
    return width(e);
  }
  static std::streamsize width(dispatch<false>, const C& e) {
    std::streamsize w = width(e);
    if(min(e) < 0) return w+1;
    else return w;
  }
};


};

#endif /* TVMET_IO_H */

// Local Variables:
// mode:C++
// End:
