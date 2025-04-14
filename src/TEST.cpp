#include <iostream>
#include <xsimd/xsimd.hpp>
#include <array>
#include <cstdint>

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



int main() {
    using batch = xsimd::batch<int32_t>;
    constexpr std::size_t BATCH_SIZE = batch::size;
    

    std::array<int32_t, BATCH_SIZE> data{};
    for (int i = 0; i < BATCH_SIZE; ++i)
        data[i] = i + 1;

    batch original = batch::load_unaligned(data.data());

    std::array<int32_t, BATCH_SIZE> original_vals;
    original.store_unaligned(original_vals.data());

    std::cout << "Original: ";
    for (int i = 0; i < BATCH_SIZE; ++i)
        std::cout << original_vals[i] << " ";
    std::cout << std::endl;

    batch slid = slide_right_logical(original, 1);

    std::array<int32_t, BATCH_SIZE> slid_vals;
    slid.store_unaligned(slid_vals.data());

    std::cout << "Slide Right <1>: ";
    for (int i = 0; i < BATCH_SIZE; ++i)
        std::cout << slid_vals[i] << " ";
    std::cout << std::endl;

    slid_vals[0] = 99;
    batch with_insert = batch::load_unaligned(slid_vals.data());

    std::cout << "After insert(0, 99): ";
    for (size_t i = 0; i < BATCH_SIZE; ++i)
        std::cout << slid_vals[i] << " ";
    std::cout << std::endl;

    return 0;
}
