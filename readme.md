# a_star
This is a naive implementation of [A* search](https://en.wikipedia.org/wiki/A*_search_algorithm) in Nvidia CUDA.

## Example
Finding the shortest distance from A to Z in the following graph:

![](https://serope.com/ai/a-star-sm.png)

```
nodes 
                    -
                    -
                    -
                    -
                    -
                    -
                    -
                    -
visited_nodes 
                    A (0x5010a0000) 
                    B (0x5010a0200) 
                    C (0x5010a0400) 
                    D (0x5010a0600) 
                    E (0x5010a0800) 
                    F (0x5010a0a00) 
                    G (0x5010a0c00) 
                    Z (0x5010a0e00) 
distances 
                    0.000 
                    10.470 
                    12.470 
                    7.470 
                    17.470 
                    14.220 
                    19.220 
                    22.770 
parents 
                    -
                    A (0x5010a0000) 
                    A (0x5010a0000) 
                    A (0x5010a0000) 
                    C (0x5010a0400) 
                    D (0x5010a0600) 
                    D (0x5010a0600) 
                    E (0x5010a0800) 


real	0m0.088s
user	0m0.004s
sys	0m0.083s
```
