/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2007 Benoit Jacob <jacob@math.jussieu.fr>
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
 */
#ifndef EIGEN_TESTSUITE_MAIN_H
#define EIGEN_TESTSUITE_MAIN_H

#include <QtTest/QtTest>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#ifdef EIGEN_USE_COMPLEX
#include <complex>
#endif

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

#include <tvmet/util/Random.h>

#include "compare.h"

using namespace tvmet;
using namespace util;
using namespace std;

class TvmetTestSuite : public QObject
{
    Q_OBJECT

  public:
    TvmetTestSuite() {};

  private slots:
    void selfTest();
    void testNumericTraits();
};

#endif // EIGEN_TESTSUITE_MAIN_H
