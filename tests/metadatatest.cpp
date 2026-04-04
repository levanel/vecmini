#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "IndexIVF.h"
#include "AlignedAllocator.h"

int main(){
    int d = 128;
    int nb = 100000;
    int nq = 1;
    int nlist = 100;
    int nprobe = 10;
    std::cout<<"running ts"<<"\n";

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float>dist(0.0, 1.0);
    AlignedVector32<float>db_v(nb*d);
    AlignedVector32<float>q_v(nq*d);
    std::vector<uint64_t>db_id(nb);
    for(int i =0; i<nb; i++){
        db_id[i]=i;
    }
    for(int i = 0; i<nb*d; i++){
        db_v[i] = dist(rng);
    }

    for(int i = 0; i<nq*d; i++){
        q_v[i] = dist(rng);
    }
    std::vector<float>distances(5);
    std::vector<int>assignids(5);
    IndexIVF ivf(d, nlist);
    ivf.train(nb, db_v.data());
    ivf.add(nb, db_v.data(), db_id.data());
    
    std::vector<uint8_t>bitmask(nb, 1);
    for(int i = 0; i<bitmask.size(); i++){
        if(i%2==0){
            bitmask[i]=0;
        }
    }
    auto start1 = std::chrono::high_resolution_clock::now();
    ivf.search(nq, q_v.data(), 5, nprobe, nullptr , distances.data(), assignids.data());
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time1 = end1 - start1;
    auto start2 = std::chrono::high_resolution_clock::now();
    ivf.search(nq, q_v.data(), 5, nprobe, bitmask.data() , distances.data(), assignids.data());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time2 = end2 - start2;
    std::cout<<"nullptr: "<< time1.count()<<"\n";
    std::cout<<"bitmask: "<< time2.count()<<"\n";
    return 0;
    
};