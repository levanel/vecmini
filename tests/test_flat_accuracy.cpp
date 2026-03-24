#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "IndexFlat.h"
#include "IndexIVF.h"

int main() {
    int d = 128;
    int nb = 100000;  // 100,000 database vectors
    int nq = 1;       // 1 query
    int nlist = 100;  // 100 Voronoi cell buckets

    std::cout << "--- VECMINI BENCHMARK ARENA ---\n";
    std::cout << "Dimensions: " << d << " | Vectors: " << nb << "\n\n";

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    std::vector<float> database(nb * d);
    std::vector<float> query(nq * d);
    for(int i = 0; i < nb * d; i++) database[i] = dist(rng);
    for(int i = 0; i < nq * d; i++) query[i] = dist(rng);

    std::vector<float> distances(nq);
    std::vector<int> labels(nq);
    
    std::cout << "FlatL2\n";
    IndexFlatL2 flat_index(d);
    flat_index.add(nb, database.data());

    auto start_flat = std::chrono::high_resolution_clock::now();
    flat_index.search(nq, query.data(), 1, distances.data(), labels.data());
    auto end_flat = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_flat = end_flat - start_flat;

    std::cout << "FlatL2 Search Time: " << diff_flat.count() << " ms\n\n";

    std::cout << "Training & Loading IVF\n";
    std::cout << "(Please wait, K-Means is drawing the map...)\n";
    IndexIVF ivf_index(d, nlist);
    ivf_index.train(nb, database.data()); // Train the map
    ivf_index.add(nb, database.data());   //File the data into drawers

    auto start_ivf = std::chrono::high_resolution_clock::now();
    ivf_index.search(nq, query.data(), 1, distances.data(), labels.data());
    auto end_ivf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_ivf = end_ivf - start_ivf;

    std::cout << "IVF Search Time:    " << diff_ivf.count() << " ms\n\n";

    float speedup = diff_flat.count() / diff_ivf.count();
    std::cout << ">>>IVF is " << speedup << "x faster<<<\n";

    return 0;
}