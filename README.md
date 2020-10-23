Description
===========

Optimized computations of Higher-Order-Marginal (HOM) of multiple sequence alignments (MSAs) and "r20" comparison scores. The code is parallelized and uses a trie for histogram counting.

Setup
=====

Compile by running "make", which should produce seqtools.xxx.so and highmarg.xxx.so.  Then you can run the `HOM_r20.py` script.  (this code isn't pip-installable).


Requirements
===========

When run this use all available CPUs, so will be faster if you run it on a computer with many CPUs. The optimizations also depend "fork" capabilities to spawn processes, which is only available on unix systems (and not osx). The code will still run on osx/windows but slower.

Instructions
============

Demo files are in the "examples" folder. Run the following commands from the examples folder.

The HOM script works in two modes, "make_db" and "count". 

First, in "make_db" mode you use it to create a "database" of the top-20 marginals at different random sets of positions, as follows:

```
$ ../HOM_r20.py make_db my_example 1000 targetMSA --npos 2-8
```

This runs the "make_db" action, to create a database named "my_example" (the name can be anything you want), it will create a database with 1000 randomly sampled sets-of-positions for each subsequence length, it will build the database of top-20 marginals for the MSA targetMSA, and it will do this for subsequence length 2 to 8 (if ommitted, the npos default is 2-10).

This should run very quickly, and create a file "top20_my_example.db". This is a text file you can look at if you want, the contents are intuitive.

Next, in "count" mode you can compute r20 scores measuring the similarity in the top 20 marginals of another MSA, as follows:

```
$ ../HOM_r20.py count my_example r20_mi modelMSA indepMSA
```

This runs the "count" action, against the database "my_example", and will write the output to the file r20_mi.npy. It computes the r20 scores for the modelMSA and the indepMSA. Alternately, multiple MSAs may be supplied, or only one (eg, just modelMSA). The output file r20_mi.npy is a numpy array of dimension NxRxM where N is the number of sets of positions (eg, npos=2-8 means 6 sets), R is the number of samples (1000 in the database constructed above) and M is the number of input MSAs (2 in this case, modelMSA and indepMSA).

Typical Usage and Tips
======================

To produce the plots like in certain publications:
```python
>>> r20 = np.load('r20_mi.npy')
>>> r20 = np.nanmean(r20, axis=1)
>>> plt.plot(range(2,9), r20[:,0], label='model')
>>> plt.plot(range(2,9), r20[:,1], label='indep')
```

The example code ran very fast because the example MSAs are so small. In practice, you want to use much larger MSAs, eg 10^6 or 10^7 sequences in order to get good estimates of the marginals. Eg, in the database you can see that 8th-order marginals are very tiny, eg 1 count (1/1000, since the example MSAs are 1000 seqs). For such poorly sampled marginals, the pearson r20 is likely to give nans eg if all the counts are 1/1000 (which is why I need nanmean above). Increasing the number of sequences will eliminate the nan problem.

The r20 you calculate from some MSAs can be low either because the "modelMSA" is mismatched to the "targetMSA", which we call "specification error", or because you used too few sequences in the r20 calculation, which we call "validation error". Validation error is uninteresting, you want to eliminate it. To test how much validation error you have, try getting another MSA drawn from the same distribution as the targetMSA, of the same size as your target/model MSAs, so that you expect the r20 to be 1. Then measure the r20. If it is not 1, it means you have validation error. We can also work out with simple assumptions, that if in this validation error test your MSAs had N0 sequenes and you gor an r20 value of r0, you can extrapolate using the formula r^2 = (N/N0) r0^2 / ((N/N0-1) r0^2 + 1) to get the expected r20 score "r" you would get if you increased the MSA size to N sequences.

This script supports variations on the r20 score through the "--topmode" and "--score" options. "--topmode" can be either an integer (default 20) specifying the number of largest subsequence marginals to track, a string like ">0.01" which says to track all subsequences above 1% frequency, or 'nonzero' to track all observed subsequences. "--score" can be "pearson", "spearman", or "pcttvd" which specifies which summary statistic to use to compare the database and MSA marginals statistics.
