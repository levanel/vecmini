#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <fstream>

// Include your vecmini headers
#include "IndexFlat.h"
#include "IndexIVF.h"
#include "IndexIVFPQ.h"

// --- THE FVECS PARSER ---
// Reads Meta's standard binary vector format directly into RAM
bool read_fvecs(const char* fname, std::vector<float>& data, int& num_vectors, int& dim) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << fname << "\n";
        return false;
    }

    // Read the dimension of the first vector
    file.read((char*)&dim, sizeof(int));
    
    // Find file size to calculate total vectors
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Each vector is 1 int (dimension) + 'dim' floats
    size_t bytes_per_vector = sizeof(int) + dim * sizeof(float);
    num_vectors = file_size / bytes_per_vector;

    data.resize(num_vectors * dim);

    // Read the binary stream into our 1D array
    for (int i = 0; i < num_vectors; i++) {
        int d;
        file.read((char*)&d, sizeof(int)); // Read dimension
        // Read the actual float data directly into our vector
        file.read((char*)(data.data() + i * dim), dim * sizeof(float)); 
    }
    
    return true;
}

int main() {
    std::cout << "--- VECMINI SIFT1M BENCHMARK ARENA ---\n";

    // --- 1. LOAD REAL DATA ---
    std::vector<float> database;
    std::vector<float> queries;
    int nb, nq, d, d_query;

    std::cout << "Loading SIFT1M Database...\n";
    if (!read_fvecs("../data/sift/sift_base.fvecs", database, nb, d)) return -1;
    nb = 500000;
    database.resize(nb * d);
    std::cout << "Loading SIFT1M Queries...\n";
    if (!read_fvecs("../data/sift/sift_query.fvecs", queries, nq, d_query)) return -1;

    std::cout << "Loaded " << nb << " vectors of dimension " << d << ".\n\n";

    // --- 2. CONFIGURATION ---
    int query_idx = 0; // We'll test the very first query in the file
    int k = 100;       // Top 100 results for Recall
    int nlist = 2048;  // 2048 Voronoi buckets
    int m = 8;         // 8 PQ Sub-spaces

    std::vector<uint64_t> database_ids(nb);
    for(int i = 0; i < nb; i++) database_ids[i] = i; 

    // Extract exactly 1 query (128 floats)
    const float* single_query = queries.data() + (query_idx * d);

    // --- 3. GET THE GROUND TRUTH (FlatL2) ---
    std::cout << "Calculating Ground Truth (FlatL2)..." << std::endl;
    IndexFlatL2 flat(d);
    flat.add(nb, database.data());
    
    std::vector<float> true_dists(k);
    std::vector<int> true_labels(k);
    flat.search(1, single_query, k, true_dists.data(), true_labels.data());

    std::unordered_set<int> ground_truth_set(true_labels.begin(), true_labels.end());

    // --- 4. SETUP IVFPQ ---
    std::cout << "\nInitializing IVFPQ Engine...\n";
    IndexIVFPQ ivfpq(d, nlist, m);
    
    ivfpq.train(nb, database.data());
    std::cout << "Populating IVFPQ Memory Drawers...\n";
    ivfpq.add(nb, database.data(), database_ids.data());

    // --- 5. SWEEP NPROBE AND CALCULATE RECALL ---
    std::vector<int> nprobe_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    
    std::vector<float> test_dists(k);
    std::vector<int64_t> test_labels(k);

    std::cout << "\n--- RECALL VS LATENCY BENCHMARK ---\n";
    std::cout << "nprobe | Latency (ms) | Recall@" << k << "\n";
    std::cout << "-----------------------------------\n";

    for (int probe : nprobe_values) {
        auto start = std::chrono::high_resolution_clock::now();
        ivfpq.search(1, single_query,k,probe, test_dists.data(), test_labels.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> diff = end - start;
        
        int matches = 0;
        for (int i = 0; i < k; i++) {
            if (ground_truth_set.count(test_labels[i]) > 0) matches++;
        }
        
        float recall = (float)matches / k * 100.0f;
        std::cout << probe << "\t | " << diff.count() << " ms \t| " << recall << "%\n";
    }

    std::cout << "-----------------------------------\n";
    return 0;
}

/*#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "IndexFlat.h"
#include "IndexIVF.h"
#include "IndexIVFPQ.h"
int main() {
    int d = 128;
    int nb = 1000000;  // 100,000 database vectors
    int nq = 1;       // 1 query
    int nlist = 100;  // 100 Voronoi cell buckets

    std::cout << "--- VECMINI BENCHMARK ARENA ---\n";
    std::cout << "Dimensions: " << d << " | Vectors: " << nb << "\n\n";

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    std::vector<float> database(nb * d);
    std::vector<float> query(nq * d);
    std::vector<uint64_t> database_ids(nb);
    for(int i = 0; i < nb; i++) {
        database_ids[i] = i; // dummy for jus givin them basic IDs 0 to 99,999
}
    for(int i = 0; i < nb * d; i++) database[i] = dist(rng);
    for(int i = 0; i < nq * d; i++) query[i] = dist(rng);





    ///temp *100 please remove it after the test!!!!!!!






    std::vector<float> distances(nq*1);
    std::vector<int> labels(nq*1);
    
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
    ivf_index.add(nb, database.data(), database_ids.data());   //File the data into drawers

    auto start_ivf = std::chrono::high_resolution_clock::now();
    ivf_index.search(nq, query.data(), 1, 1, nullptr, distances.data(), labels.data());    
    auto end_ivf = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_ivf = end_ivf - start_ivf;

    std::cout << "IVF Search Time:    " << diff_ivf.count() << " ms\n\n";

    float speedup = diff_flat.count() / diff_ivf.count();
    std::cout << ">>>IVF is " << speedup << "x faster<<<\n";

    IndexIVFPQ ivfpq(128, 2048, 8);
    ivfpq.train(nb, database.data()); 
    ivfpq.add(nb, database.data(), database_ids.data());

    auto start_ivfpq = std::chrono::high_resolution_clock::now();
    ivfpq.search(nq, query.data(), 1, 1, distances.data(), labels.data());    
    auto end_ivfpq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff_ivfpq = end_ivfpq - start_ivfpq;

    std::cout << "IVFpq Search Time:    " << diff_ivfpq.count() << " ms\n\n";

    return 0;
}
*/