#include "IndexIVF.h"
#include "clustering.h"
IndexIVF::IndexIVF(int d, int nbucket): d(d), nbucket(nbucket), router(d){
    memory.resize(nbucket);
};

void IndexIVF::train(int n, const float *x){
     if(trained) return;
     std::vector<float>centroids(nbucket*d);
     kmean_clustering(d, n, nbucket, x ,centroids.data());
     router.add(nbucket, centroids.data());
     trained=true;
}
void IndexIVF::add(int n, const float *x){
     if(!trained) return;
     std::vector<int> assign(n);
     std::vector<float> distances(n);
     router.search(n,x,1,distances.data(), assign.data());
     for(int i =0; i<n; i++){
        int bucketid= assign[i];
        memory[bucketid].insert(memory[bucketid].end(),x+(i*d), x+(i*d)+d);
    }
    ntotal+=n;
}
void IndexIVF::search(int n, const float* x, int k,float *distances, int *labels){
    std::vector<int> assign(n);
    std::vector<float> centroids_distance(n);
    router.search(n,x,1,centroids_distance.data(), assign.data());
    for(int i = 0; i<n; i++){
        int bucketid = assign[i];
        IndexFlatL2 localcheck(d);
        int vector_in_memo = memory[bucketid].size()/d;
        if(vector_in_memo==0)continue;
        localcheck.add(vector_in_memo, memory[bucketid].data());
        const float *spec_quer = x+(i*d);
        float* spec_dist=distances + (i*k);
        int* spec_lbls = labels+(i*k);
        localcheck.search(1,spec_quer, k, spec_dist, spec_lbls);
    }
}
