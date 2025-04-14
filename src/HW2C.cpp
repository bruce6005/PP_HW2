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
#include <cstring>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "ssw_cpp.h"
#include <xsimd/xsimd.hpp>
#include <cstdint>

using std::string;
using std::cout;
using std::endl;

using namespace std;
using namespace chrono;


struct AlignmentResult {
    int score;
    int ref_end;
    int query_end;
    vector<std::vector<int>> DP;
};

void PrintVisualAlignment(const std::pair<std::string, std::string>& aligned_pair, int ref_start = 0, int query_start = 0, int score = -1) {
    const std::string& ref_aligned = aligned_pair.first;
    const std::string& query_aligned = aligned_pair.second;
    std::string match_line;

    int ref_end = ref_start;
    int query_end = query_start;

    for (size_t i = 0; i < ref_aligned.size(); ++i) {
        char r = ref_aligned[i];
        char q = query_aligned[i];
        if (r != '-') ++ref_end;
        if (q != '-') ++query_end;

        if (r == q)
            match_line += '|';
        else if (r == '-' || q == '-')
            match_line += ' ';
        else
            match_line += '*';
    }

    if (score >= 0)
        std::cout << "Score: " << score << std::endl;

    std::cout << "Seq1 ref: " << std::setw(10) << ref_start << "    " << ref_aligned << "    " << (ref_end - 1) << std::endl;
    std::cout << "                        "<<std::setw(10) << match_line << std::endl;
    std::cout << "Seq2 qry: " << std::setw(10) << query_start << "    " << query_aligned << "    " << (query_end - 1) << std::endl;
}

// Profile generation: builds scoring matrix for each base 128*query
std::vector<xsimd::batch<int32_t >> build_profile(const std::string& query, int match, int mismatch) {
    using batch = xsimd::batch<int32_t >;
    int segLen = (query.size() + batch::size - 1) / batch::size;
    std::vector<batch> profile(128 * segLen);

    for (int c = 0; c < 128; ++c) {
        for (int i = 0; i < segLen; ++i) {
            std::array<int32_t, batch::size> tmp{};
            for (size_t j = 0; j < batch::size; ++j) {
                size_t idx = i + j * segLen;
                
                tmp[j] = (idx < query.size() && query[idx] == c) ? match  : mismatch ;
            }
            profile[c * segLen + i] = batch::load_unaligned(tmp.data());
        }
    }

    
    return profile;
}

void dump_profile_for_char(char c, const std::string& query, const std::vector<xsimd::batch<int32_t>>& profile) {
    using batch = xsimd::batch<int32_t>;
    const int BATCH_SIZE = batch::size;
    int segLen = (query.size() + BATCH_SIZE - 1) / BATCH_SIZE;

    std::cout << "Dumping profile for ref_char = '" << c << "' (ASCII " << int(c) << ")\n";
    for (int i = 0; i < segLen; ++i) {
        std::array<int32_t, BATCH_SIZE> tmp{};
        profile[c * segLen + i].store_unaligned(tmp.data());

        for (int j = 0; j < BATCH_SIZE; ++j) {
            int idx = i + j * segLen;
            if (idx < query.size()) {
                std::cout << "  query[" << idx << "] = " << query[idx]
                          << ", score = " << tmp[j]
                          << ((query[idx] == c) ? "  ✅" : "  ❌") << "\n";
            }
        }
    }
}
template <typename T>
xsimd::batch<T> slide_right_logical(const xsimd::batch<T>& v, std::size_t shift) {
    constexpr std::size_t N = xsimd::batch<T>::size;
    std::array<T, N> arr{};
    v.store_unaligned(arr.data());

    for (int i = N - 1; i >= static_cast<int>(shift); --i)
        arr[i] = arr[i - shift];
    for (std::size_t i = 0; i < shift; ++i)
        arr[i] = 0;

    return xsimd::batch<T>::load_unaligned(arr.data());
}
template <typename T>
xsimd::batch<T> slide_left_logical(const xsimd::batch<T>& v, std::size_t shift) {
    constexpr std::size_t N = xsimd::batch<T>::size;
    std::array<T, N> arr{};
    v.store_unaligned(arr.data());

    for (std::size_t i = 0; i + shift < N; ++i)
        arr[i] = arr[i + shift];
    for (std::size_t i = N - shift; i < N; ++i)
        arr[i] = 0;

    return xsimd::batch<T>::load_unaligned(arr.data());
}

template <typename T>
xsimd::batch<T> rotate_left_batch(const xsimd::batch<T>& v, std::size_t shift) {
    constexpr std::size_t N = xsimd::batch<T>::size;
    shift = shift % N;  // 保證合法

    std::array<T, N> input{};
    v.store_unaligned(input.data());

    std::array<T, N> rotated{};
    for (std::size_t i = 0; i < N; ++i)
        rotated[i] = input[(i + shift) % N];  // 循環左移

    return xsimd::batch<T>::load_unaligned(rotated.data());
}
AlignmentResult striped_sw(const std::string& ref, const std::string& query,
                           int match = 5, int mismatch = -2, int gap = -3) {
    using batch = xsimd::batch<int32_t>;
    constexpr int BATCH_SIZE = batch::size;
    std::cout << "batch::size = " << batch::size << std::endl;
    
    

    int refLen = ref.size();
    int queryLen = query.size();
    std::vector<std::vector<int>> DP(refLen + 1, std::vector<int>(queryLen + 1, 0));


    int segLen = (queryLen + BATCH_SIZE - 1) / BATCH_SIZE;
    cout<<segLen<<endl;

    std::vector<batch> H(segLen, batch(int32_t(0)));
    std::vector<batch> E(segLen, batch(int32_t(0)));
    // std::vector<batch> Hmax(segLen, batch(int32_t(0)));
    

    auto profile = build_profile(query, match, mismatch);

    std::vector<batch> prev_H(segLen, batch(0)); // 保存上一輪 H
    std::vector<batch> vH(segLen, batch(0)); // 保存上一輪 H
    std::vector<batch> vHStore(segLen, batch(0)); // 保存上一輪 H
    std::vector<batch> vHLoad(segLen, batch(0)); // 保存上一輪 H
    std::vector<batch> vF(segLen, batch(0)); // 上一輪H 部ROTATE
    std::vector<batch> vE(segLen, batch(0)); // 上一輪H 部ROTATE

    batch vGapO(-gap);
    batch vGapE(-gap);
    batch vZero(int32_t(0));
    batch H_prev = vZero; // 上一輪的BATCH
    batch vMax = vZero;
    int32_t max_score = 0;
    int end_ref = -1, end_query = -1;

    for (int r = 0; r < refLen; ++r) {
        
        const batch* vP = &profile[ref[r] * segLen];
        // batch h = vZero;
        // batch e = vZero;
        
        batch vF = vZero;
        std::rotate(vHStore.begin(), vHStore.begin() + 1, vHStore.end());
        vH = vHStore;
        swap(vHLoad,vHStore);
        for (int i = 0; i < segLen; ++i) {
            // 1. 斜對角 (↖)：由上一個 batch 取得 H(i-1, j-1)
            batch vH = vH +vP[i] ;
            // cout<<"順便看分數"<<vP[i]<<endl;
            // H_prev = prev_H[i]; // 更新 H_prev 為當前值（留給下一格使用）
                        
            // batch vScore = vHp + vP[i];  // match/mismatch 分數 斜對角
            vMax = xsimd::max(vMax, vH);

            vH = xsimd::max(vH, vE[i]);
            vH = xsimd::max(vH, vF);

            vHStore[i] = vH;
            vH = xsimd::max(vH - vGapO, vZero);
            vE[i] = xsimd::max(vE[i]  - vGapE, vZero);
            vE[i] = xsimd::max(vE[i]  , vH);
            vF = xsimd::max(vF  - vGapE, vZero);
            vF = xsimd::max(vF  , vH);
            vH = vHLoad[i];
            

            std::array<int32_t, BATCH_SIZE> vals;
            vH.store_unaligned(vals.data());
            for(int k=0; k<BATCH_SIZE;k++){

                if(i+segLen*k<queryLen)
                    DP[r + 1][i+segLen*k + 1] = vals[k];

                // if(k+segLen*i<queryLen)
                //     DP[r + 1][k+segLen*i + 1] = vals[k];

                // cout<<"SCORE/MAX/POS  "<<vals[k]<<" "<<max_score<<" "<<"i: "<<i<<" k:"<<k<<endl;
                if(i+segLen*k<queryLen && vals[k] > max_score ){
                        // cout<< "UPDATE :"<<i+segLen*k<<endl;
                        max_score = vals[k];
                        end_ref = r;
                        end_query = i+segLen*k;
                }
            }
        }


        // int j=0;
            // while( !xsimd::all(vF > (vHStore[j] - vGapO))){
            //     vHStore[j] = xsimd::max(vF  , vHStore[j]);
            //     vF = xsimd::max(vF-vGapE  , vZero);

            //     if(++j>=segLen){
            //         vF = slide_left_logical(vF,1);
            //         j=0;
            //     }
            // }
            
        // bool done = false;
        // int lazy_j = 0;
        // while (!done) {
        //     done = true;
        //     for (int j = 0; j < segLen; ++j) {
        //         batch h = H[j];
        //         batch f_sub = vF - vGapE;
        //         batch h_sub = h - vGapO;

        //         batch f = xsimd::max(f_sub, h_sub);
        //         batch new_h = xsimd::max(h, f);
        //         new_h = xsimd::max(new_h, vZero);

        //         // 更新 H
        //         done = done && xsimd::all(new_h == h);
        //         H[j] = new_h;
        //     }
        //     // 左移 F
        //     vF = slide_left_logical(vF, 1);
        // }
        cout<<endl;
    }

    return {max_score, end_ref, end_query,DP};
}



AlignmentResult striped_sw_scalar(const std::string& ref, const std::string& query,
                                   int match = 5, int mismatch = -2, int gap = -3) {
    int refLen = ref.size();
    int queryLen = query.size();

    std::vector<int> prev_row(queryLen + 1, 0);
    std::vector<int> curr_row(queryLen + 1, 0);

    int max_score = 0;
    int ref_end = -1;
    int query_end = -1;
    std::vector<std::vector<int>> DP(refLen + 1, std::vector<int>(queryLen + 1, 0));
    for (int i = 1; i <= refLen; ++i) {
        for (int j = 1; j <= queryLen; ++j) {
            int score_diag = prev_row[j - 1] + ((ref[i - 1] == query[j - 1]) ? match : mismatch);
            int score_up = prev_row[j] + gap;
            int score_left = curr_row[j - 1] + gap;
            int score = std::max({0, score_diag, score_up, score_left});
            curr_row[j] = score;
            DP[i][j] = score;
            if (score > max_score) {
                
                max_score = score;
                ref_end = i - 1;
                query_end = j - 1;
            }
        }
        std::swap(prev_row, curr_row);
    }

    return {max_score, ref_end, query_end,DP};
}

std::tuple<std::string, std::string, int, int> traceback(
    const std::string& ref, const std::string& query,
    const std::vector<std::vector<int>>& DP,
    int end_ref, int end_query,
    int match = 5, int mismatch = -2, int gap = -3) {
    
    std::string aligned_ref, aligned_query;
    int i = end_ref + 1;
    int j = end_query + 1;

    while (i > 0 && j > 0 && DP[i][j] > 0) {
        int score = DP[i][j];
        int diag = DP[i - 1][j - 1];
        int up = DP[i - 1][j];
        int left = DP[i][j - 1];

        if (score == diag + ((ref[i - 1] == query[j - 1]) ? match : mismatch)) {
            aligned_ref += ref[i - 1];
            aligned_query += query[j - 1];
            --i; --j;
        }
        else if (score == up + gap) {
            aligned_ref += ref[i - 1];
            aligned_query += '-';
            --i;
        }
        else {
            aligned_ref += '-';
            aligned_query += query[j - 1];
            --j;
        }
    }

    std::reverse(aligned_ref.begin(), aligned_ref.end());
    std::reverse(aligned_query.begin(), aligned_query.end());

    return {aligned_ref, aligned_query, i, j};  // 起點 i, j
}


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


void print_dp_matrix(const std::vector<std::vector<int>>& DP,
    const std::string& ref, const std::string& query) {
int refLen = ref.size();
int queryLen = query.size();

// Column label row
std::cout << std::setw(6) << ' ';
for (int j = 0; j <= queryLen; ++j) {
if (j == 0)
std::cout << std::setw(4) << '-';
else
std::cout << std::setw(4) << query[j - 1];
}
std::cout << '\n';

// DP rows with ref label
for (int i = 0; i <= refLen; ++i) {
if (i == 0)
std::cout << std::setw(4) << '-' << " ";
else
std::cout << std::setw(4) << ref[i - 1] << " ";

for (int j = 0; j <= queryLen; ++j) {
std::cout << std::setw(4) << DP[i][j];
}
std::cout << '\n';
}
}



auto run_ssw(string ref, string query, bool isSIMD){
    int querySize= query.size();
    
    

    auto start = high_resolution_clock::now();
    auto result = isSIMD? striped_sw(ref, query) : striped_sw_scalar(ref, query);
    cout<<"isSIMD "<<isSIMD<<" query_end:"<<result.query_end<<"  ref_end:"<<result.ref_end<<" "<<endl;
    print_dp_matrix(result.DP,ref,query);
    
    
    auto [aligned_ref, aligned_query, ref_start, query_start] =
    traceback(ref, query, result.DP, result.ref_end, result.query_end);
    PrintVisualAlignment({aligned_ref, aligned_query}, ref_start, query_start, result.score);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);  // 轉換成毫秒
    cout<<"SCORE: "<<result.score<<endl;
    if(isSIMD)cout << "SIMD time execution : " << duration.count() << " ns" << endl;
    else cout << "NO SIMD time execution : " << duration.count() << " ns" << endl;
    return duration.count();
}

int main() {
    string ref = read_fasta("ref.fasta");
    // int refSize= ref.size();
    string query = read_fasta("query.fasta");
    auto SIMDtime = run_ssw(ref, query,true);
    cout<<endl;
    auto basetime = run_ssw(ref, query,false);
    cout<<"Speedup:" << basetime*1.0 /SIMDtime<<"X"<<endl;
    return 0;
}

