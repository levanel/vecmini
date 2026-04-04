#pragma once
#include <cstdlib>
#include <new>
#include <vector>

// This template forces the OS to align memory to a specific byte boundary (e.g., 32 for AVX2)
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    // THE MISSING PIECE: The Rebind struct
    // This tells std::vector how to handle our custom 'Alignment' parameter
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;
    
    template <class U> 
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        // Calculate total bytes needed
        // std::size_t size = n * sizeof(T);
        // Ensure the requested size is a multiple of alignment for std::aligned_alloc
        std::size_t size = (n * sizeof(T) + Alignment - 1) & ~(Alignment - 1);
        
        // C++17 aligned allocation command to the OS
        void* ptr = std::aligned_alloc(Alignment, size);
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p); // Return the custom memory back to the OS
    }
};

// C++ STL requires custom allocators to have comparison operators
template <class T, class U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) { return true; }

template <class T, class U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) { return false; }

// A clean alias so you don't have to type that massive template everywhere
template <typename T>
using AlignedVector32 = std::vector<T, AlignedAllocator<T, 32>>;