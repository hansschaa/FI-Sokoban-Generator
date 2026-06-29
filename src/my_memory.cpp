#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include "my_memory.h"
#include <stdexcept>
 
my_memory_pool::my_memory_pool()
    : used_size(0)
    , blockSize_(0)
    , poolSize_(0)
    , numBlocks_(0)
    , pool_(nullptr)
{}

void my_memory_pool::init(size_t blockSize, size_t numBlocks){
    size_t newPoolSize = blockSize * numBlocks;
    
    if (pool_) {
        // Reuse if the new size is the same or smaller, and block size matches
        if (blockSize_ == blockSize && newPoolSize <= poolSize_) {
            numBlocks_ = numBlocks;
            used_size = 0;
            freeList_.clear();
            freeList_.reserve(numBlocks_);
            for (size_t i = 0; i < numBlocks_; ++i) {
                freeList_.push_back(pool_ + i * blockSize_);
            }
            return;
        }
        
        std::free(pool_);
        pool_ = nullptr;
        freeList_.clear();
    }

    blockSize_ = blockSize;
    numBlocks_ = numBlocks;
    poolSize_ = newPoolSize;
    used_size = 0;
    pool_ = static_cast<char*>(std::malloc(poolSize_));
    if (!pool_) {
        throw std::bad_alloc();
    }
    // FIX: antes freeList_ era std::deque, que hace una heap allocation
    // por cada push_back — exactamente lo que el pool quiere evitar.
    // Ahora es std::vector con reserve() para una sola allocación.
    freeList_.reserve(numBlocks_);
    for (size_t i = 0; i < numBlocks_; ++i) {
        freeList_.push_back(pool_ + i * blockSize_);
    }
    memset(pool_, 0, poolSize_);
}

void my_memory_pool::clear(){
    used_size = 0;
    freeList_.clear();
    // FIX: mismo reserve que en init() para evitar reallocaciones
    freeList_.reserve(numBlocks_);
    for (size_t i = 0; i < numBlocks_; ++i) {
        freeList_.push_back(pool_ + i * blockSize_);
    }
}

my_memory_pool::~my_memory_pool() {
    std::free(pool_);
}

void* my_memory_pool::allocate() {
    if (freeList_.empty()) {
        throw std::runtime_error("OOM");
    }
    void* block = freeList_.back();
    freeList_.pop_back();
    used_size += 1;
    return block;
}

void my_memory_pool::deallocate(void* block) {
    used_size -= 1;
    freeList_.push_back(static_cast<char*>(block));
}