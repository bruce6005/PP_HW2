// ==========================
// example.cpp
// This is a simple example to show you how to use the SSW C++ library.
// To run this example:
// 1) g++ -Wall ssw_cpp.cpp ssw.c example.cpp
// 2) ./a.out
// Created by Wan-Ping Lee on 09/04/12.
// Last revision by Mengyao Zhao on 2023-Apr-21
// ==========================

#include <iostream>
#include <string.h>
#include <iostream>
#include <string.h>
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "ssw_cpp.h"
#include <xsimd/xsimd.hpp>
#include <cstdint>
#include "ssw_cpp.h"
using std::string;
using std::cout;
using std::endl;

using namespace std;
using namespace chrono;
using std::string;
using std::cout;
using std::endl;

void PrintCigarAlignment(const std::string& ref, const std::string& query, const std::string& cigar,
  int ref_start = 0, int query_start = 0, int score = -1) {
std::string aligned_ref, aligned_query, match_line;
int i_ref = ref_start, i_query = query_start;

int len = 0;
for (char c : cigar) {
if (isdigit(c)) {
len = len * 10 + (c - '0');
} else {
for (int i = 0; i < len; ++i) {
char r = '-', q = '-', m = ' ';
if (c == 'M') {
if (i_ref < (int)ref.size()) r = ref[i_ref++];
if (i_query < (int)query.size()) q = query[i_query++];
m = (r == q) ? '|' : '*';
} else if (c == 'I') {
if (i_query < (int)query.size()) q = query[i_query++];
r = '-';
m = ' ';
} else if (c == 'D') {
if (i_ref < (int)ref.size()) r = ref[i_ref++];
q = '-';
m = ' ';
}
aligned_ref += r;
aligned_query += q;
match_line += m;
}
len = 0;
}
}

int ref_end = ref_start;
for (char c : aligned_ref) if (c != '-') ++ref_end;
--ref_end;

int query_end = query_start;
for (char c : aligned_query) if (c != '-') ++query_end;
--query_end;

if (score >= 0)
std::cout << "Score: " << score << "\n";

std::cout << "Seq1: " << std::setw(6) << ref_start << "    " << aligned_ref << "    " << ref_end << "\n";
std::cout << "               " << match_line << "\n";
std::cout << "Seq2: " << std::setw(6) << query_start << "    " << aligned_query << "    " << query_end << "\n";
}


static void PrintAlignment(const StripedSmithWaterman::Alignment& alignment);
string read_fasta(const string& path) {
  ifstream file(path);
  string line, seq;
  while (getline(file, line)) {
      if (line.empty()) continue;
      if (line[0] == '>') continue;
      seq += line;
  }
  return seq;
}

int main() {
  string ref = read_fasta("ref.fasta");
  string query = read_fasta("query.fasta");
  int32_t maskLen = strlen(query.c_str())/2;
  maskLen = maskLen < 15 ? 15 : maskLen;

  // Declares a default Aligner
  StripedSmithWaterman::Aligner aligner;
  // Declares a default filter
  StripedSmithWaterman::Filter filter;
  // Declares an alignment that stores the result
  StripedSmithWaterman::Alignment alignment;
  // Aligns the query to the ref
  aligner.Align(query.c_str(), ref.c_str(), ref.size(), filter, &alignment, maskLen);

  PrintAlignment(alignment);
  PrintCigarAlignment(ref,query,alignment.cigar_string,alignment.ref_begin, alignment.query_begin);
  return 0;
}

static void PrintAlignment(const StripedSmithWaterman::Alignment& alignment){
  cout << "===== SSW result =====" << endl;
  cout << "Best Smith-Waterman score:\t" << alignment.sw_score << endl
       << "Next-best Smith-Waterman score:\t" << alignment.sw_score_next_best << endl
       << "Reference start:\t" << alignment.ref_begin << endl
       << "Reference end:\t" << alignment.ref_end << endl
       << "Query start:\t" << alignment.query_begin << endl
       << "Query end:\t" << alignment.query_end << endl
       << "Next-best reference end:\t" << alignment.ref_end_next_best << endl
       << "Number of mismatches:\t" << alignment.mismatches << endl
       << "Cigar: " << alignment.cigar_string << endl;
  cout << "======================" << endl;
}
