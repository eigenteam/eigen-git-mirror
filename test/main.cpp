// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#include "main.h"

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    bool has_set_repeat = false;
    bool has_set_seed = false;
    bool need_help = false;
    unsigned int seed;
    int repeat;

    QStringList args = QCoreApplication::instance()->arguments();
    args.takeFirst(); // throw away the first argument (path to executable)
    foreach(QString arg, args)
    {
      if(arg.startsWith("r"))
      {
        if(has_set_repeat)
        {
          qDebug() << "Argument" << arg << "conflicting with a former argument";
          return 1;
        }
        repeat = arg.remove(0, 1).toInt();
        has_set_repeat = true;
        if(repeat <= 0)
        {
          qDebug() << "Invalid \'repeat\' value" << arg;
          return 1;
        }
      }
      else if(arg.startsWith("s"))
      {
        if(has_set_seed)
        {
          qDebug() << "Argument" << arg << "conflicting with a former argument";
          return 1;
        }
        bool ok;
        seed = arg.remove(0, 1).toUInt(&ok);
        has_set_seed = true;
        if(!ok)
        {
          qDebug() << "Invalid \'seed\' value" << arg;
          return 1;
        }
      }
      else
      {
        need_help = true;
      }
    }

    if(need_help)
    {
      qDebug() << "This test application takes the following optional arguments:";
      qDebug() << "  rN     Repeat each test N times (default:" << DEFAULT_REPEAT << ")";
      qDebug() << "  sN     Use N as seed for random numbers (default: based on current time)";
      return 1;
    }

    if(!has_set_seed) seed = (unsigned int) time(NULL);
    if(!has_set_repeat) repeat = DEFAULT_REPEAT;

    qDebug() << "Initializing random number generator with seed" << seed;
    srand(seed);
    qDebug() << "Repeating each test" << repeat << "times";

    Eigen::EigenTest test(repeat);
    return QTest::qExec(&test, 1, argv);
}

#include "main.moc"
