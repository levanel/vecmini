#include "IndexFlat.h"

void IndexFlatL2::add(int n, const float *x){
    xb.insert(xb.end(), x, x+(n*d));
    ntotal+=n;
}