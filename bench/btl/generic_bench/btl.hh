//=====================================================
// File   :  btl.hh
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
//=====================================================
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
#ifndef BTL_HH
#define BTL_HH

#include "bench_parameter.hh"
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include "utilities.h"

/** Enhanced std::string
*/
class BtlString : public std::string
{
public:
    BtlString() : std::string() {}
    BtlString(const BtlString& str) : std::string(static_cast<const std::string&>(str)) {}
    BtlString(const std::string& str) : std::string(str) {}
    BtlString(const char* str) : std::string(str) {}

    operator const char* () const { return c_str(); }

    void trim( bool left = true, bool right = true )
    {
        int lspaces, rspaces, len = length(), i;
        lspaces = rspaces = 0;

        if ( left )
            for (i=0; i<len && (at(i)==' '||at(i)=='\t'||at(i)=='\r'||at(i)=='\n'); ++lspaces,++i);

        if ( right && lspaces < len )
            for(i=len-1; i>=0 && (at(i)==' '||at(i)=='\t'||at(i)=='\r'||at(i)=='\n'); rspaces++,i--);

        *this = substr(lspaces, len-lspaces-rspaces);
    }

    std::vector<BtlString> split( const BtlString& delims = "\t\n ") const
    {
        std::vector<BtlString> ret;
        unsigned int numSplits = 0;
        size_t start, pos;
        start = 0;
        do
        {
            pos = find_first_of(delims, start);
            if (pos == start)
            {
                ret.push_back("");
                start = pos + 1;
            }
            else if (pos == npos)
                ret.push_back( substr(start) );
            else
            {
                ret.push_back( substr(start, pos - start) );
                start = pos + 1;
            }
            //start = find_first_not_of(delims, start);
            ++numSplits;
        } while (pos != npos);
        return ret;
    }

    bool endsWith(const BtlString& str) const
    {
        if(str.size()>this->size())
            return false;
        return this->substr(this->size()-str.size(),str.size()) == str;
    }
    bool contains(const BtlString& str) const
    {
        return this->find(str)<this->size();
    }
    bool beginsWith(const BtlString& str) const
    {
        if(str.size()>this->size())
            return false;
        return this->substr(0,str.size()) == str;
    }

    BtlString toLowerCase( void )
    {
        std::transform(begin(), end(), begin(), static_cast<int(*)(int)>(::tolower) );
        return *this;
    }
    BtlString toUpperCase( void )
    {
        std::transform(begin(), end(), begin(), static_cast<int(*)(int)>(::toupper) );
        return *this;
    }

    /** Case insensitive comparison.
    */
    bool isEquiv(const BtlString& str) const
    {
        BtlString str0 = *this;
        str0.toLowerCase();
        BtlString str1 = str;
        str1.toLowerCase();
        return str0 == str1;
    }

    /** Decompose the current string as a path and a file.
        For instance: "dir1/dir2/file.ext" leads to path="dir1/dir2/" and filename="file.ext"
    */
    void decomposePathAndFile(BtlString& path, BtlString& filename) const
    {
        std::vector<BtlString> elements = this->split("/\\");
        path = "";
        filename = elements.back();
        elements.pop_back();
        if (this->at(0)=='/')
            path = "/";
        for (unsigned int i=0 ; i<elements.size() ; ++i)
            path += elements[i] + "/";
    }
};

class BtlConfig
{
public:
  BtlConfig()
    : m_runSingleAction(false)
  {
    char * _config;
    _config = getenv ("BTL_CONFIG");
    if (_config!=NULL)
    {
      std::vector<BtlString> config = BtlString(_config).split(" \t\n");
      for (int i = 0; i<config.size(); i++)
      {
        if (config[i].beginsWith("-a"))
        {
          if (i+1==config.size())
          {
            std::cerr << "error processing option: " << config[i] << "\n";
            exit(2);
          }
          Instance.m_runSingleAction = true;
          Instance.m_singleActionName = config[i+1];

          i += 1;
        }
      }
    }
  }

  static bool skipAction(const std::string& name)
  {
    if (Instance.m_runSingleAction)
    {
      std::cout << "Instance.m_singleActionName = " << Instance.m_singleActionName << "\n";
      return !BtlString(name).contains(Instance.m_singleActionName);
    }

    return false;
  }


protected:
  bool m_runSingleAction;
  BtlString m_singleActionName;

  static BtlConfig Instance;
};

#define BTL_MAIN \
  BtlConfig BtlConfig::Instance

#endif // BTL_HH
