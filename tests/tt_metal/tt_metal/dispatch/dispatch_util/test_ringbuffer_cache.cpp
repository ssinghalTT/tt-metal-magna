// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include <memory>
#include <numeric>
#include <vector>
#include <algorithm>

namespace tt::tt_metal {
class RingbufferCacheTestFixture : public ::testing::Test {
protected:
    RingbufferCacheTestFixture() = default;
    ~RingbufferCacheTestFixture() override = default;

    // Define cache of size 256 blocks, one block of size 4 unsigned.
    constexpr static size_t cache_block_sizeB = 4;
    constexpr static size_t cache_size_blocks = 256;
    constexpr static size_t initial_manager_size = 64;  // initial size of the manager array, less than cache size
    std::unique_ptr<RingbufferCacheManager> rbCache;

    // This function is called before each test in this test case.
    void SetUp() override {
        rbCache = std::make_unique<RingbufferCacheManager>(cache_block_sizeB, cache_size_blocks, initial_manager_size);
    }

    // This function is called after each test in this test case.
    void TearDown() override {}

    // define accessors to the private members of the RingbufferCacheManager
    auto get_next_block_offset() const { return rbCache->manager_.next_block_offset; }
    auto get_oldest_idx() const { return rbCache->manager_.oldest_idx; }
    auto get_next_idx() const { return rbCache->manager_.next_idx; }
    auto get_manager_entry_size() const { return rbCache->manager_.entry.size(); }
    auto get_manager_entry(size_t idx) const { return rbCache->manager_.entry[idx]; }
    auto get_valid_entry(size_t idx) const { return rbCache->valid_[idx]; }
    constexpr static auto invalid_entry_ = RingbufferCacheManager::invalid_cache_entry_;
};

TEST_F(RingbufferCacheTestFixture, SimpleAllocate) {
    auto result = rbCache->get_cache_offset(0, 10);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->is_cached, false);
    EXPECT_EQ(result->offset, 0);
}

TEST_F(RingbufferCacheTestFixture, AllocateMoreThanCacheSize) {
    // Try to allocate more than 256 blocks, expect failure
    auto result = rbCache->get_cache_offset(0, (cache_size_blocks + 1) * cache_block_sizeB);
    ASSERT_FALSE(result);
}

TEST_F(RingbufferCacheTestFixture, ConfirmCachedBlocks) {
    // Allocate blocks of sizes {2, 4, 8, 16, 32, 64, 128}, confirm they are cached
    std::vector<size_t> block_sizes = {2, 4, 8, 16, 32, 64, 128};
    std::vector<size_t> pgm_ids;
    auto last_offset = 0;
    for (auto i = 0; i < block_sizes.size(); ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        auto result = rbCache->get_cache_offset(i, pgm_size);
        pgm_ids.push_back(i);
        ASSERT_TRUE(result);
        if (result) {
            EXPECT_EQ(result->is_cached, false);
            EXPECT_EQ(result->offset, last_offset);
            last_offset += block_sizes[i];
        }
    }
    last_offset = 0;
    // cache hits
    for (auto i = 0; i < block_sizes.size(); ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        auto result = rbCache->get_cache_offset(pgm_ids[i], pgm_size);
        ASSERT_TRUE(result);
        if (result) {
            EXPECT_EQ(result->is_cached, true);
            EXPECT_EQ(result->offset, last_offset);
            last_offset += block_sizes[i];
        }
    }
    EXPECT_EQ(get_manager_entry_size(), initial_manager_size);
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_idx(), block_sizes.size());
}

TEST_F(RingbufferCacheTestFixture, WraparoundAllocateLargeBlock) {
    // Allocate blocks of sizes {2, 4, 8, 16, 32, 64, 128, 128, 8}, confirm wraparound allocation and eviction of
    // previous entries
    std::vector<size_t> block_sizes = {2, 4, 8, 16, 32, 64, 128, 128};
    std::vector<size_t> pgm_ids;
    size_t last_offset = 0;
    int i;
    for (i = 0; i < block_sizes.size() - 1; ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        auto result = rbCache->get_cache_offset(i, pgm_size);
        EXPECT_EQ(get_oldest_idx(), 0);
        pgm_ids.push_back(i);
        ASSERT_TRUE(result);
        if (result) {
            EXPECT_EQ(result->is_cached, false);
            EXPECT_EQ(result->offset, last_offset);
            last_offset += block_sizes[i];
        }
    }
    EXPECT_EQ(get_next_block_offset(), std::reduce(block_sizes.begin(), block_sizes.end() - 1));
    // Allocate last block to invalidate all previous entries. Then check everything is invalidated. Check internal
    // manager state is as expected
    auto pgm_size = block_sizes[i] * cache_block_sizeB;
    auto result = rbCache->get_cache_offset(i, pgm_size);
    pgm_ids.push_back(i);
    ASSERT_TRUE(result);
    if (result) {
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, 0);
    }

    for (i = 0; i < block_sizes.size() - 1; ++i) {
        EXPECT_EQ(get_valid_entry(i), invalid_entry_);
    }
    EXPECT_EQ(get_oldest_idx(), 7);
    EXPECT_EQ(get_next_block_offset(), block_sizes.back() & (cache_size_blocks - 1));

    EXPECT_EQ(get_manager_entry_size(), initial_manager_size);
    EXPECT_EQ(get_next_idx(), block_sizes.size());
}

TEST_F(RingbufferCacheTestFixture, EvictOldestEntries) {
    // Allocate blocks of sizes {48, 48, 128}, then allocate 64 and confirm eviction
    std::vector<size_t> block_sizes = {48, 48, 128};
    std::vector<size_t> pgm_ids;
    size_t last_offset = 0;
    int i;
    for (i = 0; i < block_sizes.size(); ++i) {
        auto pgm_size = block_sizes[i] * cache_block_sizeB;
        rbCache->get_cache_offset(i, pgm_size);
        pgm_ids.push_back(i);
    }
    rbCache->get_cache_offset(i, 64 * cache_block_sizeB);
    // pgm_ids.push_back(i);

    // Confirm evictions
    for (auto it_pgm_id = pgm_ids.begin(); it_pgm_id != pgm_ids.end() - 1; ++it_pgm_id) {
        EXPECT_EQ(get_valid_entry(*it_pgm_id), invalid_entry_);
    }

    // Confirm oldest_idx
    EXPECT_EQ(get_oldest_idx(), 2);
}

TEST_F(RingbufferCacheTestFixture, ComplexEvictionAndAllocation) {
    // Allocate blocks of sizes {128, 64, 32}, then allocate 64, 40, and 50
    std::vector<size_t> block_sizes = {128, 64, 32};
    int last_offset = 0;
    for (auto i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
    }
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_block_offset(), std::reduce(block_sizes.begin(), block_sizes.end()));

    auto result = rbCache->get_cache_offset(block_sizes.size(), 64 * cache_block_sizeB);  // Evict first block
    EXPECT_EQ(result->offset, 0);
    EXPECT_EQ(get_valid_entry(0), invalid_entry_);

    result = rbCache->get_cache_offset(block_sizes.size() + 1, 40 * cache_block_sizeB);  // No eviction
    EXPECT_EQ(result->offset, 64);
    EXPECT_EQ(get_valid_entry(1), 1);

    result = rbCache->get_cache_offset(block_sizes.size() + 2, 50 * cache_block_sizeB);  // Evict second block
    EXPECT_EQ(result->offset, (64 + 40));
    EXPECT_EQ(get_valid_entry(1), invalid_entry_);
    EXPECT_EQ(get_valid_entry(2), 2);
    EXPECT_EQ(get_oldest_idx(), 2);
    EXPECT_EQ(get_next_block_offset(), 64 + 40 + 50);
}

TEST_F(RingbufferCacheTestFixture, ValidateCacheFullBehavior) {
    // Validate behavior when cache is completely full
    std::vector<size_t> block_sizes(256, 1);  // Fill cache with 256 blocks of size 1
    int last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i + 100, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 100), result->offset);
    }

    // Allocate a new block to trigger eviction
    rbCache->get_cache_offset(10, 1 * cache_block_sizeB);

    EXPECT_EQ(get_manager_entry_size(), std::min(block_sizes.size() * 2, cache_size_blocks));
    // Validate the new block is added at the correct offset
    EXPECT_EQ(get_manager_entry(0).offset, 0);
    EXPECT_EQ(get_manager_entry(0).valid_idx, 10);
    EXPECT_EQ(get_valid_entry(10), 0);

    EXPECT_EQ(get_next_idx(), 1);
    EXPECT_EQ(get_oldest_idx(), 1);
}

TEST_F(RingbufferCacheTestFixture, FillResetRefill) {
    // Fill the cache
    std::vector<size_t> block_sizes(4, 64);  // Fill cache with 256 blocks of size 1
    int last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i + 1000, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 1000), i);
    }

    // Reset the cache
    rbCache->reset();

    // Refill the cache
    last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(i + 10, block_sizes[i] * cache_block_sizeB);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->is_cached, false);
        EXPECT_EQ(result->offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 10), i);
    }
    // Validate the cache is filled correctly
    last_offset = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        EXPECT_EQ(get_manager_entry(i).valid_idx, i + 10);
        EXPECT_EQ(get_manager_entry(i).offset, last_offset);
        last_offset += block_sizes[i];
        EXPECT_EQ(get_valid_entry(i + 10), i);
    }
    EXPECT_EQ(get_next_block_offset(), 0);
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_idx(), block_sizes.size());
}

TEST_F(RingbufferCacheTestFixture, ValidateWraparoundAllocate) {
    // fillup cache
    int i;
    for (i = 0; i < cache_size_blocks; ++i) {
        rbCache->get_cache_offset(i, 1 * cache_block_sizeB);
    }
    EXPECT_EQ(get_manager_entry_size(), cache_size_blocks);
    EXPECT_EQ(get_oldest_idx(), 0);
    EXPECT_EQ(get_next_block_offset(), 0);
    auto next_idx = 0;
    EXPECT_EQ(get_next_idx(), 0);

    std::vector<size_t> block_sizes{1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89};  // first 11 fibonacci
    std::reverse(block_sizes.begin(), block_sizes.end());
    auto fibonacci_total = std::reduce(block_sizes.begin(), block_sizes.end());

    // Fill the cache
    int last_offset = 0;
    for (i = 0; i < block_sizes.size(); ++i) {
        auto result = rbCache->get_cache_offset(cache_size_blocks + i, block_sizes[i] * cache_block_sizeB);
    }
    EXPECT_EQ(get_next_block_offset(), fibonacci_total);
    next_idx = block_sizes.size();
    EXPECT_EQ(get_next_idx(), next_idx);
    EXPECT_EQ(
        get_oldest_idx(),
        fibonacci_total);  // because had to free up fib total number of 1 block allocations that were made before

    // Allocate a new block to trigger wraparound
    auto eviction_blocks_num = 48;
    auto result = rbCache->get_cache_offset(cache_size_blocks + i, eviction_blocks_num * cache_block_sizeB);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->is_cached, false);
    EXPECT_EQ(result->offset, 0);
    // Check current state:
    //  next_idx was at end of fibonacci sequence, i.e 11, so now it must be 12. Since the cache wrapped around, the
    //  older entries from fib total to end of cache (single block allocations) must have been evicted. That would take
    //  oldest_idx all the way to the end of the manager and wrapped to the front of the manager array. Finally, since
    //  the next block was allocated to cache[0], oldest_idx must've incremented to 1.
    EXPECT_EQ(get_oldest_idx(), 1);
    EXPECT_EQ(get_next_idx(), 12);
    // verify that oldest_idx is pointing to cache block at higher address than next_block_offset
    EXPECT_GE(get_manager_entry(get_oldest_idx()).offset, get_next_block_offset());
    EXPECT_EQ(get_valid_entry(get_manager_entry(get_oldest_idx()).valid_idx), get_oldest_idx());
    EXPECT_EQ(get_next_block_offset(), eviction_blocks_num);

    // The last allocation, as explained above, must have been at offset 11. Let's confirm it is correct.
    auto current_idx = get_next_idx() - 1;
    EXPECT_EQ(get_manager_entry(current_idx).offset, 0);
    EXPECT_EQ(get_manager_entry(current_idx).valid_idx, cache_size_blocks + i);
    EXPECT_EQ(get_valid_entry(cache_size_blocks + i), current_idx);

    for (i = fibonacci_total; i < block_sizes.size(); ++i) {
        EXPECT_EQ(get_valid_entry(i), invalid_entry_);
    }
}

}  // namespace tt::tt_metal
