#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>

using namespace std;

struct inputfile_entry_t
{
  uint16_t product_size;
  uint16_t block_size;
  float gflops;
};

struct inputfile_t
{
  string filename;
  vector<inputfile_entry_t> entries;

  inputfile_t(const string& fname)
    : filename(fname)
  {
    ifstream stream(filename);
    if (!stream.is_open()) {
      cerr << "couldn't open input file: " << filename << endl;
      exit(1);
    }
    string line;
    bool is_in_measurements = false;
    while (getline(stream, line)) {
      if (line.empty()) continue;
      if (line.find("BEGIN MEASUREMENTS") == 0) {
        is_in_measurements = true;
        continue;
      }

      if (!is_in_measurements) {
        continue;
      }

      unsigned int product_size, block_size;
      float gflops;
      int sscanf_result =
        sscanf(line.c_str(), "%x %x %f",
               &product_size,
               &block_size,
               &gflops);
      if (3 != sscanf_result ||
          !product_size ||
          product_size > 0xfff ||
          !block_size ||
          block_size > 0xfff ||
          !isfinite(gflops))
      {
        cerr << "ill-formed input file: " << filename << endl;
        cerr << "offending line:" << endl << line << endl;
        exit(1);
      }
      inputfile_entry_t entry;
      entry.product_size = uint16_t(product_size);
      entry.block_size = uint16_t(block_size);
      entry.gflops = gflops;
      entries.push_back(entry);
    }
    stream.close();
    if (!is_in_measurements) {
      cerr << "Input file " << filename << " didn't contain a BEGIN MEASUREMENTS line. Wrong file?" << endl;
      exit(1);
    }
    if (entries.empty()) {
      cerr << "didn't find any measurements in input file: " << filename << endl;
      exit(1);
    }
    //cerr << "read " << entries.size() << " measurements from " << filename << endl;
  }
};

struct preprocessed_inputfile_entry_t
{
  uint16_t product_size;
  uint16_t block_size;

  float efficiency;
};

struct preprocessed_inputfile_t
{
  string filename;
  vector<preprocessed_inputfile_entry_t> entries;

  preprocessed_inputfile_t(const inputfile_t& inputfile)
    : filename(inputfile.filename)
  {
    auto it = inputfile.entries.begin();
    auto it_first_with_given_product_size = it;
    while (it != inputfile.entries.end()) {
      ++it;
      if (it == inputfile.entries.end() ||
        it->product_size != it_first_with_given_product_size->product_size)
      {
        import_input_file_range_one_product_size(it_first_with_given_product_size, it);
        it_first_with_given_product_size = it;
      }
    }
  }

private:
  void import_input_file_range_one_product_size(
    const vector<inputfile_entry_t>::const_iterator& begin,
    const vector<inputfile_entry_t>::const_iterator& end)
  {
    uint16_t product_size = begin->product_size;
    float max_gflops = 0.0f;
    for (auto it = begin; it != end; ++it) {
      if (it->product_size != product_size) {
        cerr << "Unexpected ordering of entries in " << filename << endl;
        cerr << "(Expected all entries for product size " << hex << product_size << dec << " to be grouped)" << endl;
        exit(1);
      }
      max_gflops = max(max_gflops, it->gflops);
    }
    for (auto it = begin; it != end; ++it) {
      preprocessed_inputfile_entry_t entry;
      entry.product_size = it->product_size;
      entry.block_size = it->block_size;
      entry.efficiency = it->gflops / max_gflops;
      entries.push_back(entry);
    }
  }
};

void check_all_files_in_same_exact_order(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles)
{
  if (preprocessed_inputfiles.empty()) {
    return;
  }

  const preprocessed_inputfile_t& first_file = preprocessed_inputfiles[0];
  const size_t num_entries = first_file.entries.size();

  for (size_t i = 0; i < preprocessed_inputfiles.size(); i++) {
    if (preprocessed_inputfiles[i].entries.size() != num_entries) {
      cerr << "these files have different number of entries: "
           << preprocessed_inputfiles[i].filename
           << " and "
           << first_file.filename
           << endl;
      exit(1);
    }
  }

  for (size_t entry_index = 0; entry_index < num_entries; entry_index++) {
    const uint16_t entry_product_size = first_file.entries[entry_index].product_size;
    const uint16_t entry_block_size = first_file.entries[entry_index].block_size;
    for (size_t file_index = 0; file_index < preprocessed_inputfiles.size(); file_index++) {
      const preprocessed_inputfile_t& cur_file = preprocessed_inputfiles[file_index];
      if (cur_file.entries[entry_index].product_size != entry_product_size ||
          cur_file.entries[entry_index].block_size != entry_block_size)
      {
        cerr << "entries not in same order between these files: "
             << first_file.filename
             << " and "
             << cur_file.filename
             << endl;
        exit(1);
      }
    }
  }
}

float efficiency_of_subset(
        const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
        const vector<size_t>& subset)
{
  if (subset.size() <= 1) {
    return 1.0f;
  }
  const preprocessed_inputfile_t& first_file = preprocessed_inputfiles[subset[0]];
  const size_t num_entries = first_file.entries.size();
  float efficiency = 1.0f;
  size_t entry_index = 0;
  size_t first_entry_index_with_this_product_size = 0;
  uint16_t product_size = first_file.entries[0].product_size;
  while (entry_index < num_entries) {
    ++entry_index;
    if (entry_index == num_entries ||
        first_file.entries[entry_index].product_size != product_size)
    {
      float efficiency_this_product_size = 0.0f;
      for (size_t e = first_entry_index_with_this_product_size; e < entry_index; e++) {
        float efficiency_this_entry = 1.0f;
        for (auto i = subset.begin(); i != subset.end(); ++i) {
          efficiency_this_entry = min(efficiency_this_entry, preprocessed_inputfiles[*i].entries[e].efficiency);
        }
        efficiency_this_product_size = max(efficiency_this_product_size, efficiency_this_entry);
      }
      efficiency = min(efficiency, efficiency_this_product_size);
      first_entry_index_with_this_product_size = entry_index;
      product_size = first_file.entries[entry_index].product_size;
    }
  }

  return efficiency;
}

float efficiency_of_partition(
        const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
        const vector<vector<size_t>>& partition)
{
  float efficiency = 1.0f;
  for (auto s = partition.begin(); s != partition.end(); ++s) {
    efficiency = min(efficiency, efficiency_of_subset(preprocessed_inputfiles, *s));
  }
  return efficiency;
}

void make_first_subset(size_t subset_size, vector<size_t>& out_subset, size_t set_size)
{
  assert(subset_size >= 1 && subset_size <= set_size);
  out_subset.resize(subset_size);
  for (size_t i = 0; i < subset_size; i++) {
    out_subset[i] = i;
  }
}

bool is_last_subset(const vector<size_t>& subset, size_t set_size)
{
  return subset[0] == set_size - subset.size();
}

void next_subset(vector<size_t>& inout_subset, size_t set_size)
{
  if (is_last_subset(inout_subset, set_size)) {
    cerr << "iterating past the last subset" << endl;
    abort();
  }
  size_t i = 1;
  while (inout_subset[inout_subset.size() - i] == set_size - i) {
    i++;
    assert(i <= inout_subset.size());
  }
  size_t first_index_to_change = inout_subset.size() - i;
  inout_subset[first_index_to_change]++;
  size_t p = inout_subset[first_index_to_change];
  for (size_t j = first_index_to_change + 1; j < inout_subset.size(); j++) {
    inout_subset[j] = ++p;
  }
}

const size_t number_of_subsets_limit = 100;
const size_t always_search_subsets_of_size_at_least = 2;

bool is_number_of_subsets_feasible(size_t n, size_t p)
{ 
  assert(n>0 && p>0 && p<=n);
  uint64_t numerator = 1, denominator = 1;
  for (size_t i = 0; i < p; i++) {
    numerator *= n - i;
    denominator *= i + 1;
    if (numerator > denominator * number_of_subsets_limit) {
      return false;
    }
  }
  return true;
}

size_t max_feasible_subset_size(size_t n)
{
  assert(n > 0);
  const size_t minresult = min<size_t>(n-1, always_search_subsets_of_size_at_least);
  for (size_t p = 1; p <= n - 1; p++) {
    if (!is_number_of_subsets_feasible(n, p+1)) {
      return max(p, minresult);
    }
  }
  return n - 1;
}

void find_subset_with_efficiency_higher_than(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
       float required_efficiency_to_beat,
       vector<size_t>& inout_remainder,
       vector<size_t>& out_subset)
{
  out_subset.resize(0);

  if (required_efficiency_to_beat >= 1.0f) {
    cerr << "can't beat efficiency 1." << endl;
    abort();
  }

  while (!inout_remainder.empty()) {

    vector<size_t> candidate_indices(inout_remainder.size());
    for (size_t i = 0; i < candidate_indices.size(); i++) {
      candidate_indices[i] = i;
    }

    size_t candidate_indices_subset_size = max_feasible_subset_size(candidate_indices.size());
    while (candidate_indices_subset_size >= 1) {
      vector<size_t> candidate_indices_subset;
      make_first_subset(candidate_indices_subset_size,
                        candidate_indices_subset,
                        candidate_indices.size());

      vector<size_t> best_candidate_indices_subset;
      float best_efficiency = 0.0f;
      vector<size_t> trial_subset = out_subset;
      trial_subset.resize(out_subset.size() + candidate_indices_subset_size);
      while (true)
      {
        for (size_t i = 0; i < candidate_indices_subset_size; i++) {
          trial_subset[out_subset.size() + i] = inout_remainder[candidate_indices_subset[i]];
        }
        
        float trial_efficiency = efficiency_of_subset(preprocessed_inputfiles, trial_subset);
        if (trial_efficiency > best_efficiency) {
          best_efficiency = trial_efficiency;
          best_candidate_indices_subset = candidate_indices_subset;
        }
        if (is_last_subset(candidate_indices_subset, candidate_indices.size())) {
          break;
        }
        next_subset(candidate_indices_subset, candidate_indices.size());
      }
       
      if (best_efficiency > required_efficiency_to_beat) {
        for (size_t i = 0; i < best_candidate_indices_subset.size(); i++) {
          candidate_indices[i] = candidate_indices[best_candidate_indices_subset[i]];
        }
        candidate_indices.resize(best_candidate_indices_subset.size());
      }
      candidate_indices_subset_size--;
    }
      
    size_t candidate_index = candidate_indices[0];
    auto candidate_iterator = inout_remainder.begin() + candidate_index;
    vector<size_t> trial_subset = out_subset;

    trial_subset.push_back(*candidate_iterator);
    float trial_efficiency = efficiency_of_subset(preprocessed_inputfiles, trial_subset);
    if (trial_efficiency > required_efficiency_to_beat) {
      out_subset.push_back(*candidate_iterator);
      inout_remainder.erase(candidate_iterator);
    } else {
      break;
    }
  }
}

void find_partition_with_efficiency_higher_than(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
       float required_efficiency_to_beat,
       vector<vector<size_t>>& out_partition)
{
  out_partition.resize(0);

  vector<size_t> remainder;
  for (size_t i = 0; i < preprocessed_inputfiles.size(); i++) {
    remainder.push_back(i);
  }

  while (!remainder.empty()) {
    vector<size_t> new_subset;
    find_subset_with_efficiency_higher_than(
      preprocessed_inputfiles,
      required_efficiency_to_beat,
      remainder,
      new_subset);
    out_partition.push_back(new_subset);
  }
}

void print_partition(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
       const vector<vector<size_t>>& partition)
{
  float efficiency = efficiency_of_partition(preprocessed_inputfiles, partition);
  cout << "Partition into " << partition.size() << " subsets for " << efficiency * 100.0f << "% efficiency"  << endl;
  for (auto subset = partition.begin(); subset != partition.end(); ++subset) {
    cout << "  Subset " << (subset - partition.begin())
         << ", efficiency " << efficiency_of_subset(preprocessed_inputfiles, *subset) * 100.0f << "%:"
         << endl;
    for (auto file = subset->begin(); file != subset->end(); ++file) {
      cout << "    " << preprocessed_inputfiles[*file].filename << endl;
    }
  }
  cout << endl;
}

int main(int argc, char* argv[])
{
  if (argc == 1) {
    cerr << "usage: " << argv[0] << " [input files]" << endl;
    cerr << "the input files should each contain an output of benchmark-blocking-sizes" << endl;
    exit(1);
  }
  cout.precision(3);
  cerr.precision(3);
  vector<string> inputfilenames;
  for (int i = 1; i < argc; i++) {
  	inputfilenames.emplace_back(argv[i]);
  }

  vector<preprocessed_inputfile_t> preprocessed_inputfiles;
  for (auto it = inputfilenames.begin(); it != inputfilenames.end(); ++it) {
    preprocessed_inputfiles.emplace_back(inputfile_t(*it));
  }

  check_all_files_in_same_exact_order(preprocessed_inputfiles);

  float required_efficiency_to_beat = 0.0f;
  vector<vector<vector<size_t>>> partitions;
  cerr << "searching for partitions...\r" << flush;
  while (true)
  {
    vector<vector<size_t>> partition;
    find_partition_with_efficiency_higher_than(
      preprocessed_inputfiles,
      required_efficiency_to_beat,
      partition);
    float actual_efficiency = efficiency_of_partition(preprocessed_inputfiles, partition);
    cerr << "partition " << preprocessed_inputfiles.size() << " files into " << partition.size()
         << " subsets for " << 100.0f * actual_efficiency
         << " % efficiency"
         << "                  \r" << flush;
    partitions.push_back(partition);
    if (partition.size() == preprocessed_inputfiles.size() || actual_efficiency == 1.0f) {
      break;
    }
    required_efficiency_to_beat = actual_efficiency;
  }
  cerr << "                                                                  " << endl;
  while (true) {
    bool repeat = false;
    for (size_t i = 0; i < partitions.size() - 1; i++) {
      if (partitions[i].size() >= partitions[i+1].size()) {
        partitions.erase(partitions.begin() + i);
        repeat = true;
        break;
      }
    }
    if (!repeat) {
      break;
    }
  }
  for (auto it = partitions.begin(); it != partitions.end(); ++it) {
    print_partition(preprocessed_inputfiles, *it);
  }
}
