// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Jacob <benoitjacob@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <memory>

bool eigen_use_specific_block_size;
int eigen_block_size_k, eigen_block_size_m, eigen_block_size_n;
#define EIGEN_TEST_SPECIFIC_BLOCKING_SIZES eigen_use_specific_block_size
#define EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_K eigen_block_size_k
#define EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_M eigen_block_size_m
#define EIGEN_TEST_SPECIFIC_BLOCKING_SIZE_N eigen_block_size_n
#include <Eigen/Core>

#include <bench/BenchTimer.h>

using namespace Eigen;
using namespace std;

static BenchTimer timer;

// how many times we repeat each measurement.
// measurements are randomly shuffled - we're not doing
// all N identical measurements in a row.
const int measurement_repetitions = 3;

// Timings below this value are too short to be accurate,
// we'll repeat measurements with more iterations until
// we get a timing above that threshold.
const float min_accurate_time = 1e-2f;

// See --min-working-set-size command line parameter.
size_t min_working_set_size = 0;

// range of sizes that we will benchmark (in all 3 K,M,N dimensions)
const size_t maxsize = 2048;
const size_t minsize = 16;

typedef MatrixXf MatrixType;

static_assert((maxsize & (maxsize - 1)) == 0, "maxsize must be a power of two");
static_assert((minsize & (minsize - 1)) == 0, "minsize must be a power of two");
static_assert(maxsize > minsize, "maxsize must be larger than minsize");
static_assert(maxsize < (minsize << 16), "maxsize must be less than (minsize<<16)");

// just a helper to store a triple of K,M,N sizes for matrix product
struct size_triple_t
{
  size_t k, m, n;
  size_triple_t() : k(0), m(0), n(0) {}
  size_triple_t(size_t _k, size_t _m, size_t _n) : k(_k), m(_m), n(_n) {}
  size_triple_t(const size_triple_t& o) : k(o.k), m(o.m), n(o.n) {}
  size_triple_t(uint16_t compact)
  {
    k = 1 << ((compact & 0xf00) >> 8);
    m = 1 << ((compact & 0x0f0) >> 4);
    n = 1 << ((compact & 0x00f) >> 0);
  }
};

uint8_t log2_pot(size_t x) {
  size_t l = 0;
  while (x >>= 1) l++;
  return l;
}

// Convert between size tripes and a compact form fitting in 12 bits
// where each size, which must be a POT, is encoded as its log2, on 4 bits
// so the largest representable size is 2^15 == 32k  ... big enough.
uint16_t compact_size_triple(size_t k, size_t m, size_t n)
{
  return (log2_pot(k) << 8) | (log2_pot(m) << 4) | log2_pot(n);
}

uint16_t compact_size_triple(const size_triple_t& t)
{
  return compact_size_triple(t.k, t.m, t.n);
}

// A single benchmark. Initially only contains benchmark params.
// Then call run(), which stores the result in the gflops field.
struct benchmark_t
{
  uint16_t compact_product_size;
  uint16_t compact_block_size;
  bool use_default_block_size;
  float gflops;
  benchmark_t()
    : compact_product_size(0)
    , compact_block_size(0)
    , gflops(0)
    , use_default_block_size(false)
  {}
  benchmark_t(size_t pk, size_t pm, size_t pn,
              size_t bk, size_t bm, size_t bn)
    : compact_product_size(compact_size_triple(pk, pm, pn))
    , compact_block_size(compact_size_triple(bk, bm, bn))
    , use_default_block_size(false)
    , gflops(0)
  {}
  benchmark_t(size_t pk, size_t pm, size_t pn)
    : compact_product_size(compact_size_triple(pk, pm, pn))
    , compact_block_size(0)
    , use_default_block_size(true)
    , gflops(0)
  {}

  void run();
};

ostream& operator<<(ostream& s, const benchmark_t& b)
{
  s << hex << b.compact_product_size << dec;
  if (b.use_default_block_size) {
    s << " default";
  } else {
    s << " " << hex << b.compact_block_size << dec;
  }
  s << " " << b.gflops;
  return s;
}

// We sort first by increasing benchmark parameters,
// then by decreasing performance.
bool operator<(const benchmark_t& b1, const benchmark_t& b2)
{ 
  return b1.compact_product_size < b2.compact_product_size ||
           (b1.compact_product_size == b2.compact_product_size && (
             (b1.compact_block_size < b2.compact_block_size || (
               b1.compact_block_size == b2.compact_block_size &&
                 b1.gflops > b2.gflops))));
}

void benchmark_t::run()
{
  size_triple_t productsizes(compact_product_size);

  if (use_default_block_size) {
    eigen_use_specific_block_size = false;
  } else {
    // feed eigen with our custom blocking params
    eigen_use_specific_block_size = true;
    size_triple_t blocksizes(compact_block_size);
    eigen_block_size_k = blocksizes.k;
    eigen_block_size_m = blocksizes.m;
    eigen_block_size_n = blocksizes.n;
  }

  // set up the matrix pool

  const size_t combined_three_matrices_sizes =
    sizeof(MatrixType::Scalar) *
      (productsizes.k * productsizes.m +
       productsizes.k * productsizes.n +
       productsizes.m * productsizes.n);

  // 64 M is large enough that nobody has a cache bigger than that,
  // while still being small enough that everybody has this much RAM,
  // so conveniently we don't need to special-case platforms here.
  const size_t unlikely_large_cache_size = 64 << 20;

  const size_t working_set_size =
    min_working_set_size ? min_working_set_size : unlikely_large_cache_size;

  const size_t matrix_pool_size =
    1 + working_set_size / combined_three_matrices_sizes;

  MatrixType *lhs = new MatrixType[matrix_pool_size];
  MatrixType *rhs = new MatrixType[matrix_pool_size];
  MatrixType *dst = new MatrixType[matrix_pool_size];
  
  for (size_t i = 0; i < matrix_pool_size; i++) {
    lhs[i] = MatrixType::Zero(productsizes.m, productsizes.k);
    rhs[i] = MatrixType::Zero(productsizes.k, productsizes.n);
    dst[i] = MatrixType::Zero(productsizes.m, productsizes.n);
  }

  // main benchmark loop

  int iters_at_a_time = 1;
  float time_per_iter = 0.0f;
  size_t matrix_index = 0;
  while (true) {

    double starttime = timer.getCpuTime();
    for (int i = 0; i < iters_at_a_time; i++) {
      dst[matrix_index] = lhs[matrix_index] * rhs[matrix_index];
      matrix_index++;
      if (matrix_index == matrix_pool_size) {
        matrix_index = 0;
      }
    }
    double endtime = timer.getCpuTime();

    const float timing = float(endtime - starttime);

    if (timing >= min_accurate_time) {
      time_per_iter = timing / iters_at_a_time;
      break;
    }

    iters_at_a_time *= 2;
  }

  delete[] lhs;
  delete[] rhs;
  delete[] dst;

  gflops = 2e-9 * productsizes.k * productsizes.m * productsizes.n / time_per_iter;
}

void print_cpuinfo()
{
#ifdef __linux__
  cout << "contents of /proc/cpuinfo:" << endl;
  string line;
  ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    while (getline(cpuinfo, line)) {
      cout << line << endl;
    }
    cpuinfo.close();
  }
  cout << endl;
#elif defined __APPLE__
  cout << "output of sysctl hw:" << endl;
  system("sysctl hw");
  cout << endl;
#endif
}

template <typename T>
string type_name()
{
  return "unknown";
}

template<>
string type_name<float>()
{
  return "float";
}

template<>
string type_name<double>()
{
  return "double";
}

struct action_t
{
  virtual const char* invokation_name() const { abort(); return nullptr; }
  virtual void run() const { abort(); }
  virtual ~action_t() {}
};

void show_usage_and_exit(int argc, char* argv[],
                         const vector<unique_ptr<action_t>>& available_actions)
{
  cerr << "usage: " << argv[0] << " <action> [options...]" << endl << endl;
  cerr << "available actions:" << endl << endl;
  for (auto it = available_actions.begin(); it != available_actions.end(); ++it) {
    cerr << "  " << (*it)->invokation_name() << endl;
  }
  cerr << endl;
  cerr << "options:" << endl << endl;
  cerr << "  --min-working-set-size=N:" << endl;
  cerr << "       Set the minimum working set size to N bytes." << endl;
  cerr << "       This is rounded up as needed to a multiple of matrix size." << endl;
  cerr << "       A larger working set lowers the chance of a warm cache." << endl;
  cerr << "       The default value 0 means use a large enough working" << endl;
  cerr << "       set to likely outsize caches." << endl;
  cerr << "       A value of 1 (that is, 1 byte) would mean don't do anything to" << endl;
  cerr << "       avoid warm caches." << endl;
  exit(1);
}

void run_benchmarks(vector<benchmark_t>& benchmarks)
{
  // randomly shuffling benchmarks allows us to get accurate enough progress info,
  // as now the cheap/expensive benchmarks are randomly mixed so they average out.
  random_shuffle(benchmarks.begin(), benchmarks.end());

  // timings here are only used to display progress info.
  // Whence the use of real time.
  double time_start = timer.getRealTime();
  double time_last_progress_update = time_start;
  for (size_t i = 0; i < benchmarks.size(); i++) {
    // Display progress info on stderr
    double time_now = timer.getRealTime();
    if (time_now > time_last_progress_update + 1.0f) {
      time_last_progress_update = time_now;
      float ratio_done = float(i) / benchmarks.size();
      cerr.precision(3);
      cerr << "Measurements... " << 100.0f * ratio_done
           << " %";

      if (i > 10) {
        cerr << ", ETA ";
        int eta = int(float(time_now - time_start) * (1.0f - ratio_done) / ratio_done);
        int eta_remainder = eta;
        if (eta_remainder > 3600) {
          int hours = eta_remainder / 3600;
          cerr << hours << " h ";
          eta_remainder -= hours * 3600;
        }
        if (eta_remainder > 60) {
          int minutes = eta_remainder / 60;
          cerr << minutes << " min ";
          eta_remainder -= minutes * 60;
        }
        if (eta < 600 && eta_remainder) {
          cerr << eta_remainder << " s";
        }
      }
      cerr << "                                                \r" << flush;
    }

    // This is where we actually run a benchmark!
    benchmarks[i].run();
  }

  // Erase progress info
  cerr << "                                                            " << endl;

  // Sort timings by increasing benchmark parameters, and decreasing gflops.
  // The latter is very important. It means that we can ignore all but the first
  // benchmark with given parameters.
  sort(benchmarks.begin(), benchmarks.end());

  // Collect best (i.e. now first) results for each parameter values.
  vector<benchmark_t> best_benchmarks;
  for (auto it = benchmarks.begin(); it != benchmarks.end(); ++it) {
    if (best_benchmarks.empty() ||
        best_benchmarks.back().compact_product_size != it->compact_product_size ||
        best_benchmarks.back().compact_block_size != it->compact_block_size)
    {
      best_benchmarks.push_back(*it);
    }
  }

  // keep and return only the best benchmarks
  benchmarks = best_benchmarks;
}

struct measure_all_pot_sizes_action_t : action_t
{
  virtual const char* invokation_name() const { return "measure-all-pot-sizes"; }
  virtual void run() const
  {
    vector<benchmark_t> benchmarks;
    for (int repetition = 0; repetition < measurement_repetitions; repetition++) {
      for (size_t ksize = minsize; ksize <= maxsize; ksize *= 2) {
        for (size_t msize = minsize; msize <= maxsize; msize *= 2) {
          for (size_t nsize = minsize; nsize <= maxsize; nsize *= 2) {
            for (size_t kblock = minsize; kblock <= ksize; kblock *= 2) {
              for (size_t mblock = minsize; mblock <= msize; mblock *= 2) {
                for (size_t nblock = minsize; nblock <= nsize; nblock *= 2) {
                  benchmarks.emplace_back(ksize, msize, nsize, kblock, mblock, nblock);
                }
              }
            }
          }
        }
      }
    }

    run_benchmarks(benchmarks);

    cout << "BEGIN MEASUREMENTS ALL POT SIZES" << endl;
    for (auto it = benchmarks.begin(); it != benchmarks.end(); ++it) {
      cout << *it << endl;
    }
  }
};

struct measure_default_sizes_action_t : action_t
{
  virtual const char* invokation_name() const { return "measure-default-sizes"; }
  virtual void run() const
  {
    vector<benchmark_t> benchmarks;
    for (int repetition = 0; repetition < measurement_repetitions; repetition++) {
      for (size_t ksize = minsize; ksize <= maxsize; ksize *= 2) {
        for (size_t msize = minsize; msize <= maxsize; msize *= 2) {
          for (size_t nsize = minsize; nsize <= maxsize; nsize *= 2) {
            benchmarks.emplace_back(ksize, msize, nsize);
          }
        }
      }
    }

    run_benchmarks(benchmarks);

    cout << "BEGIN MEASUREMENTS DEFAULT SIZES" << endl;
    for (auto it = benchmarks.begin(); it != benchmarks.end(); ++it) {
      cout << *it << endl;
    }
  }
};

int main(int argc, char* argv[])
{
  cout.precision(4);
  cerr.precision(4);

  vector<unique_ptr<action_t>> available_actions;
  available_actions.emplace_back(new measure_all_pot_sizes_action_t);
  available_actions.emplace_back(new measure_default_sizes_action_t);

  auto action = available_actions.end();

  if (argc <= 1) {
    show_usage_and_exit(argc, argv, available_actions);
  }
  for (auto it = available_actions.begin(); it != available_actions.end(); ++it) {
    if (!strcmp(argv[1], (*it)->invokation_name())) {
      action = it;
      break;
    }
  }

  if (action == available_actions.end()) {
    show_usage_and_exit(argc, argv, available_actions);
  }

  for (int i = 2; i < argc; i++) {
    if (argv[i] == strstr(argv[i], "--min-working-set-size=")) {
      const char* equals_sign = strchr(argv[i], '=');
      min_working_set_size = strtoul(equals_sign+1, nullptr, 10);
    } else {
      cerr << "unrecognized option: " << argv[i] << endl << endl;
      show_usage_and_exit(argc, argv, available_actions);
    }
  }

  print_cpuinfo();

  cout << "benchmark parameters:" << endl;
  cout << "pointer size: " << 8*sizeof(void*) << " bits" << endl;
  cout << "scalar type: " << type_name<MatrixType::Scalar>() << endl;
  cout << "packet size: " << internal::packet_traits<MatrixType::Scalar>::size << endl;
  cout << "minsize = " << minsize << endl;
  cout << "maxsize = " << maxsize << endl;
  cout << "measurement_repetitions = " << measurement_repetitions << endl;
  cout << "min_accurate_time = " << min_accurate_time << endl;
  cout << "min_working_set_size = " << min_working_set_size;
  if (min_working_set_size == 0) {
    cout << " (try to outsize caches)";
  }
  cout << endl << endl;

  (*action)->run();
}
