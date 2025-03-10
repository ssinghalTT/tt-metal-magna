#include <iostream>
#include <vector>
#include "bfloat16.hpp"

// this follows the implementation of numpy's is_close
inline bool bos_is_close(float a, float b, float rtol = 0.01f, float atol = 0.001f) {
    // the idea is near zero we want absolute tolerance since relative doesn't make sense
    // (consider 1e-6f and 1.1e-6f)
    // elsewhere (not near zero) we want relative tolerance
    auto absdiff = fabsf(a - b);
    auto reldenom = fmaxf(fabsf(a), fabsf(b));
    auto result = (absdiff <= atol) || (absdiff <= rtol * reldenom);
    
    if (result != true) {
        std::cout << "Discrepacy: Host = " << a << " Device = " << b << std::endl;
        std::cout << "   absdiff = " << absdiff << std::endl;
        std::cout << "   reldiff = " << absdiff / (reldenom + 1e-6f) << std::endl;
    }
    return result;
}

inline std::vector<std::uint32_t> bos_create_linear_vector_of_bfloat16(uint32_t num_bytes, float root_value, float step) {
    const uint32_t num_elements_vec = std::max(
        static_cast<uint32_t>(1), static_cast<uint32_t>(num_bytes / sizeof(std::uint32_t)));  // always at least have 1
    std::vector<std::uint32_t> vec(num_elements_vec, 0);
    
    for (int i = 0; i < vec.size(); i++) {
        bfloat16 num_1_bfloat16 = bfloat16(root_value + step*2*i);
        bfloat16 num_2_bfloat16 = num_elements_vec == 1 ? bfloat16(static_cast<float>(0.0)) : bfloat16(root_value + step*(2*i+1));     
        vec.at(i) = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(num_1_bfloat16, num_2_bfloat16));
    }
    return vec;
}

inline std::vector<uint32_t> bos_casting_bfloat16_vec_to_uint16_vec( 
    const std::vector<std::uint32_t>& data,
    std::function<bfloat16(const bfloat16&)> transform) {
    std::vector<uint32_t> result; 
    uint16_t max_uint16 = std::numeric_limits<uint16_t>::max();
    for (auto i = 0; i < data.size(); i++) {
        uint32_t tmp_dt = 0;
        auto unpacked = unpack_two_bfloat16_from_uint32(data[i]);
        float f1 = transform(unpacked.first).to_float();
        float f2 = transform(unpacked.second).to_float();
        uint16_t d1 = static_cast<uint16_t>(f1);
        uint16_t d2 = static_cast<uint16_t>(f2);
        
        if (f1 > (float)max_uint16) {
            d1 = max_uint16;
        }
        if (f2 > (float)max_uint16) {
            d2 = max_uint16;
        }
        // if (i==29520) {
        //     std::cout << "Value = " << f1 << " And : " << f2 << std::endl;
        //     std::cout << "Value = " << d1 << " And : " << d2 << std::endl;

        // } 
        
        tmp_dt = ((uint32_t)d2<<16) | (uint32_t)d1 ;
        result.push_back(tmp_dt);
    }
    return result;
}

inline bool bos_uint16_vector_comparison(
    const std::vector<uint32_t>& vec_a,
    const std::vector<uint32_t>& vec_b,
    std::function<bool(float, float)> comparison_function,
    int* argfail = nullptr) {
    if (vec_a.size() != vec_b.size()) {
        std::cout << "Sizes don't match, returning false" << std::endl;
        std::cout << "Sizes A: " << vec_a.size() << ", Sizes B: " << vec_b.size() << std::endl;
        return false;
    }
    uint32_t num_print = 32;
    std::cout << "Output comparision: " << num_print << " elements" << std::endl;

    for (int i = 0; i < vec_a.size(); i++) {
        float a1 = static_cast<float>(vec_a[i] & 0xffff);  // lower 16
        float a2 = static_cast<float>((vec_a[i] >> 16) & 0xffff);    // upper 16
        float b1 = static_cast<float>(vec_b[i] & 0xffff);  // lower 16
        float b2 = static_cast<float>((vec_b[i] >> 16) & 0xffff);    // upper 16

        if (i < num_print/2){
            std::cout << "Host: " << std::hex << a1 << ", Device: " << b1 << std::endl;
            std::cout << "Host: " << std::hex << a2 << ", Device: " << b2 << std::endl;
        }

        if (not(comparison_function(a1, b1) and comparison_function(a2, b2))) {
            if (argfail) {
                *argfail = i;
                std::cout << "a1 = " << std::hex << a1 << std::endl;
                std::cout << "b1 = " << std::hex << b1 << std::endl;
                std::cout << "a2 = " << std::hex << a2 << std::endl;
                std::cout << "b2 = " << std::hex << b2 << std::endl;
            }
            return false;
        }
    }

    return true;
}

inline std::vector<uint32_t> bos_casting_bfloat16_vec_to_uint8_vec(
    const std::vector<std::uint32_t>& data,
    std::function<bfloat16(const bfloat16&)> transform) {
    std::vector<uint32_t> result;
    
    for (auto i = 0; i < data.size(); i+=2) {
        uint32_t tmp_dt = 0;

        auto unpacked1 = unpack_two_bfloat16_from_uint32(data[i]);
        float f1 = transform(unpacked1.first).to_float();
        float f2 = transform(unpacked1.second).to_float();
        uint32_t a1 = static_cast<uint8_t>(std::round(f1));
        uint32_t a2 = static_cast<uint8_t>(std::round(f2));

        auto unpacked2 = unpack_two_bfloat16_from_uint32(data[i+1]);
        float f3 = transform(unpacked2.first).to_float();
        float f4 = transform(unpacked2.second).to_float();
        uint32_t a3 = static_cast<uint8_t>(std::round(f3));
        uint32_t a4 = static_cast<uint8_t>(std::round(f4));
        tmp_dt = ((uint32_t)a1<<24) | ((uint32_t)a2<<16) | ((uint32_t)a3<<8) | (uint32_t)a4;
        result.push_back(tmp_dt);
    }
    return result;
}

inline bool bos_uint8_vector_comparison(
    const std::vector<uint32_t>& vec_a,
    const std::vector<uint32_t>& vec_b,
    std::function<bool(float, float)> comparison_function,
    int* argfail = nullptr) {
    if (vec_a.size() != vec_b.size()) {
        std::cout << "Sizes don't match, returning false" << std::endl;
        std::cout << "Sizes A: " << vec_a.size() << ", Sizes B: " << vec_b.size() << std::endl;
        return false;
    }

    uint32_t num_print = 32;
    std::cout << "Output comparision: " << num_print << " elements" << std::endl;

    for (int i = 0; i < vec_a.size(); i++) {
        
        float a1 = static_cast<float>((vec_a[i] >> 24) & 0xff);    // upper 16
        float a2 = static_cast<float>((vec_a[i] >> 16) & 0xff);    // upper 16
        float a3 = static_cast<float>((vec_a[i] >> 8) & 0xff);    // upper 16
        float a4 = static_cast<float>(vec_a[i] & 0xff);  // lower 16
        
        float b1 = static_cast<float>((vec_b[i] >> 24) & 0xff);    // upper 16
        float b2 = static_cast<float>((vec_b[i] >> 16) & 0xff);    // upper 16
        float b3 = static_cast<float>((vec_b[i] >> 8) & 0xff);    // upper 16
        float b4 = static_cast<float>(vec_b[i] & 0xff);  // lower 16

        if (i < num_print/4){
            std::cout << "Host: " << std::hex << a1 << ", Device: " << b1 << std::endl;
            std::cout << "Host: " << std::hex << a2 << ", Device: " << b2 << std::endl;
            std::cout << "Host: " << std::hex << a3 << ", Device: " << b3 << std::endl;
            std::cout << "Host: " << std::hex << a4 << ", Device: " << b4 << std::endl;
        }

        if (not(comparison_function(a1, b1) and comparison_function(a2, b2) and comparison_function(a3, b3) and comparison_function(a4, b4))) {
            if (argfail) {
                *argfail = i;
                std::cout << "a1 = " << std::hex << a1 << std::endl;
                std::cout << "b1 = " << std::hex << b1 << std::endl;
                std::cout << "a2 = " << std::hex << a2 << std::endl;
                std::cout << "b2 = " << std::hex << b2 << std::endl;
            }
            return false;
        }
    }

    return true;
}