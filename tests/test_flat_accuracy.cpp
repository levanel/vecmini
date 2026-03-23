#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "IndexFlat.h"

int main() {
    int d = 128;             // OpenAI-style dimensions
    int nb = 1000000;         // 100,000 database vectors
    int nq = 1;              // 1 query vector

    IndexFlatL2 index(d);

    // Set up the random number generator
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    // Allocate memory for our massive arrays
    std::vector<float> database(nb * d);
    std::vector<float> query(nq * d);

    std::cout << "Generating random data...\n";

    // TODO 1: Write a for-loop to fill the 'database' vector with random floats
    for(int i = 0; i<database.size(); i++){
        database[i] = dist(rng);
    }
    // TODO 2: Write a for-loop to fill the 'query' vector with random floats
    for(int i = 0; i<query.size(); i++){
        query[i] = dist(rng);
    }
    std::cout << "Loading database into engine...\n";
    index.add(nb, database.data());

    std::vector<float> distances(nq);
    std::vector<int> labels(nq);

    std::cout << "Firing search engine...\n";

    // TODO 3: Start the chrono stopwatch
    auto start = std::chrono::high_resolution_clock::now();
    // TODO 4: Call your index.search() function
    
    index.search(1, query.data(), 1, distances.data(), labels.data());

    // TODO 5: Stop the chrono stopwatch
    auto end = std::chrono::high_resolution_clock::now();
    // TODO 6: Calculate the time taken in milliseconds and std::cout it!
    std::chrono::duration<double, std::milli> diff=end-start;
    std::cout<<diff.count();
    return 0;
}