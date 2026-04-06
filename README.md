# vecmini

small experiment with vector search. mostly focused on comparing brute force (FlatL2) vs IVF and trying to push latency down.


24/03/26

FlatL2 Search Time: 52.3925 ms
IVF Search Time:    0.681871 ms
IVF is ~76.8x faster

---

29/03/26

Dimensions: 128
Vectors: 100000

FlatL2 Search Time: 54.944 ms

IVF Search Time:    0.439938 ms
IVF is ~124.9x faster

---

02/04/26

IVF Search Time: 0.061043 ms
still about 2x behind faiss

---
5/4/26
## recall vs latency (SIFT1M)

### before simd

```
nprobe | latency (ms) | recall@100
----------------------------------
1      | 0.123482     | 19%
2      | 0.069703     | 27%
4      | 0.09303      | 34%
8      | 0.138358     | 41%
16     | 0.22643      | 46%
32     | 0.435398     | 46%
64     | 0.75388      | 48%
128    | 1.4382       | 49%
256    | 2.815        | 49%
512    | 5.64083      | 49%
```

### after simd

```
nprobe | latency (ms) | recall@100
----------------------------------
1      | 0.109373     | 19%
2      | 0.063347     | 27%
4      | 0.074871     | 34%
8      | 0.10637      | 41%
16     | 0.15966      | 46%
32     | 0.277833     | 46%
64     | 0.491692     | 48%
128    | 0.933444     | 49%
256    | 1.79481      | 49%
512    | 3.67183      | 49%
```

---

We doin gooood
