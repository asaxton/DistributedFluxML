# jlDistributedFluxML



## Getting started
Comming soon
### Opt Out All Reduce
```
using Distributed
p = addProc(3)

@everywhere using DistributedFluxML

DistributedFluxML.OptOutAllReduce.init(p)

mock_vals = [1,2,3]

allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
            for (v,w) in zip(mock_vals,p)]
[fetch(fut) for fut in allR_fut] # [(6,3), (6,3), (6,3)]

mock_vals = [1,:Skip,3]

allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
            for (v,w) in zip(mock_vals,p)]
[fetch(fut) for fut in allR_fut] # [(4,2), (4,2), (4,2)]
```
