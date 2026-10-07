## Model Pipeline
```
Counting
      ↓
  Raw graph construction (DAG)
      ↓
  Semantic validation 
      ↓
  Semantic analysis -- 
      ↓
  Semantic optimization
      - folding
      - CSE
      - simplification
      - DCE
      ↓
  Semantic revalidation and reanalysis
      ↓
  Fusion candidate discovery
      ↓
  Region formation
      - map regions
      - producer/reduction regions
      - sibling-reduction regions
      - contraction regions
      ↓
  Region optimization
      - expression CSE
      - load deduplication
      - accumulator sharing
      - dead stores
      ↓
  Layout planning
      ↓
  Kernel planning
      ↓
  Final executable validation
      ↓
  Lifetime analysis
      ↓
  Memory planning
      ↓
  Model creation
  ```