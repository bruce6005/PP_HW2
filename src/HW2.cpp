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
    std::cout << "                 "<<std::setw(10) << match_line << std::endl;
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
                           int match = 2, int mismatch = -1, int gap = -2) {
    using batch = xsimd::batch<int32_t>;
    constexpr int BATCH_SIZE = batch::size;
    
    

    int refLen = ref.size();
    int queryLen = query.size();
    std::vector<std::vector<int>> DP(refLen + 1, std::vector<int>(queryLen + 1, 0));


    int segLen = (queryLen + BATCH_SIZE - 1) / BATCH_SIZE;

    std::vector<batch> H(segLen, batch(int32_t(0)));
    std::vector<batch> Hcp(segLen, batch(int32_t(0)));
    std::vector<batch> E(segLen, batch(int32_t(0)));
    std::vector<batch> Hmax(segLen, batch(int32_t(0)));

    auto profile = build_profile(query, match, mismatch);

    std::vector<batch> prev_H(segLen, batch(0)); // 保存上一輪 H
    std::vector<batch> vF(segLen, batch(0)); // 上一輪H 部ROTATE
    std::vector<batch> vZeroVec(segLen, batch(0)); // 上一輪H 部ROTATE

    batch vGapO(-gap);
    batch vGapE(-gap);
    batch vZero(int32_t(0));
    batch H_prev = vZero; // 上一輪的BATCH
    batch new_E = vZero;
    int32_t max_score = 0;
    int end_ref = -1, end_query = -1;

    for (int r = 0; r < refLen; ++r) {
        
        const batch* vP = &profile[ref[r] * segLen];
        
        batch h = vZero;
        batch e = vZero;
        prev_H = H;
        vF=Hcp;
        // std::rotate(prev_H.begin(), prev_H.end() - 1, prev_H.end());
        // batch H_prev = i == 0 ? prev_H.back() : prev_H[i - 1];

        for (int i = 0; i < segLen; ++i) {
            
            H_prev = i == 0 ? H[segLen-1] : H[i-1]; // 不要ROTATE 讓整個快一點
            
            batch H_prev = i == 0 ? prev_H.back() : prev_H[i - 1];
            // 1. 斜對角 (↖)：由上一個 batch 取得 H(i-1, j-1)
            // H_prev= prev_H[i];
            batch vHp = H_prev;
                        
            batch vScore = vHp + vP[i];  // match/mismatch 分數
            batch e_sub = xsimd::max(e - vGapE, vZero);
            new_E = e_sub;

            batch f_sub = xsimd::max(vF[i] - vGapE, vZero);
            batch new_F =f_sub;


            

            h = vZero;
            h = xsimd::max(vScore, new_E);
            h = xsimd::max(h, new_F);
            h = xsimd::max(h, vZero);
            Hcp[i]=h;
            if(i==segLen-1)h=slide_right_logical(h,1);
            H[i] = h;
            
            e=H[i];// 這個地方有問題


            
            std::array<int32_t, BATCH_SIZE> vals;
            h.store_unaligned(vals.data());
    
            for (int k = 0; k < BATCH_SIZE; ++k) {
                int query_pos = i + segLen * k;
                if (query_pos < queryLen) {
                    DP[r + 1][query_pos + 1] = vals[k];  // SW 傳統 DP 表填這格
    
                    if (vals[k] > max_score) {
                        max_score = vals[k];
                        end_ref = r;          // 現在是在第 r row
                        end_query = query_pos;
                    }
                }
            }
        }
// LAZY P 論文的 但看起來是錯的 但我沒時間了

        // batch vF_local = slide_left_logical(new_E, 1);  // 初始 carry，模擬 << 1
        // batch h_gap_open = H[0];
        // int j = 0;

        // while (!xsimd::any(vF_local > (h_gap_open - vGapO))) {
        //     // 判斷是否有任一 slot 滿足 propagate 條件
        //     batch h_gap_open = H[j];  // 模擬 vHStore[j]
        //     H[j] = xsimd::max(H[j], vF_local);

        //     vF_local = vF_local - vGapE;
        //     if (++j >= segLen) {
        //         vF_local = slide_left_logical(vF_local, 1);  // 再 propagate 一段
        //         j = 0;
        //     }
        // }


//LAZY P 看不懂 寫一個最爛的處理 拿出來再放回去  慢得要死
        
        std::vector<int> linear_H(queryLen, 0);  
        for (size_t i = 0; i < H.size(); ++i) {
            std::array<int32_t, BATCH_SIZE> values;
            Hcp[i].store_unaligned(values.data());  // 把 SIMD 值存到陣列
            for (int j = 0; j < BATCH_SIZE; ++j) {
                int linear_index = i + segLen * j;
                int val = values[j];
                if (linear_index < queryLen)
                    linear_H[linear_index] = val;
            }
        }
        std::vector<int> lazy_H = linear_H;
        for (int i = 1; i < queryLen; ++i) {
            int propagated = max(lazy_H[i - 1] +gap , 0);
            if (propagated > lazy_H[i]) {
                lazy_H[i] = propagated;
            }
        }
        for (int i = 0; i < queryLen; ++i) {
            if(i<queryLen){
                DP[r + 1][i + 1] = lazy_H[i];
                // DP[r + 1][i + 1] = H[i];
                if(DP[r + 1][i + 1] > max_score ){
                        max_score = DP[r + 1][i + 1];
                        end_ref = r;
                        end_query = i;
                }
            }
        }
        
        for (int i = 0; i < segLen; ++i) {
            std::array<int32_t, BATCH_SIZE> values{};
            for (int j = 0; j < BATCH_SIZE; ++j) {
                int linear_index = i + segLen * j;
                if (linear_index < queryLen){
                    values[j] = lazy_H[linear_index];
                }
                else
                    values[j] = 0;  // padding
            }
            
            xsimd::batch<int32_t> h_batch = xsimd::load_unaligned(values.data());
            Hcp[i] = h_batch;
            // 如果是最後一個 batch，做 logical 右移一格
            if (i == segLen - 1) {
                h_batch = slide_right_logical(h_batch, 1);
            }
        
            H[i] = h_batch;  // 丟回 H 中
        }
        
    }

    return {max_score, end_ref, end_query, DP};
}



AlignmentResult striped_sw_scalar(const std::string& ref, const std::string& query,
                                   int match = 2, int mismatch = -1, int gap = -2) {
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
AlignmentResult striped_sw_scalar_banded(const std::string& ref, const std::string& query,
    int match = 2, int mismatch = -1, int gap = -2,
    int band_width = 10) {
int refLen = ref.size();
int queryLen = query.size();

    std::vector<int> prev_row(queryLen + 1, 0);
    std::vector<int> curr_row(queryLen + 1, 0);

    int max_score = 0;
    int ref_end = -1;
    int query_end = -1;
    std::vector<std::vector<int>> DP(refLen + 1, std::vector<int>(queryLen + 1, 0));

    for (int i = 1; i <= refLen; ++i) {
    // 限制 j 在 band 範圍內
    int j_min = std::max(1, i - band_width);
    int j_max = std::min(queryLen, i + band_width);

    for (int j = j_min; j <= j_max; ++j) {
    int score_diag = prev_row[j - 1] + ((ref[i - 1] == query[j - 1]) ? match : mismatch);
    int score_up   = prev_row[j] + gap;
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
    std::fill(curr_row.begin(), curr_row.end(), 0);  // 清空 curr_row 下一輪使用
    }

    return {max_score, ref_end, query_end, DP};
}


std::tuple<std::string, std::string, int, int> traceback(
    const std::string& ref, const std::string& query,
    const std::vector<std::vector<int>>& DP,
    int end_ref, int end_query,
    int match = 2, int mismatch = -1, int gap = -2) {
    
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
    auto result = isSIMD? striped_sw(ref, query) : striped_sw_scalar_banded(ref, query);
    cout<<"isSIMD "<<isSIMD<<" query_end:"<<result.query_end<<"  ref_end:"<<result.ref_end<<" "<<endl;
    // print_dp_matrix(result.DP,ref,query);
    
    
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

