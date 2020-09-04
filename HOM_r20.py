#!/usr/bin/env python
import numpy as np
from numpy import random
import sys, os, argparse, warnings
import seqload
from scipy.stats import pearsonr
from multiprocessing import Pool, set_start_method, Lock
from highmarg import highmarg, countref
set_start_method('fork')

alpha = "-ACDEFGHIKLMNPQRSTVWY"

def get_subseq(msa, pos, out=None):
    """
    Create a new msa composed of only the columns in "pos" 
    This will run much faster if msa is in F order
    """
    if out is None:
        ret = np.empty((msa.shape[0], len(pos)), dtype='u1', order='F')
    else:
        assert(out.shape == (msa.shape[0], len(pos)))
        assert(out.dtype == np.dtype('u1'))
        ret = out

    for n,p in enumerate(pos):
        ret[:,n] = msa[:,p]
    return ret

###############################################################################
# functions to generate db

def process_marg(name, npos, i, pos, f, uniq, lock):
    top20 = np.argsort(f)[-20:]

    seqs = ["".join(alpha[c] for c in si) for si in uniq[top20]]
    f = f[top20]
    
    out = [" ".join(str(p) for p in pos)]
    out += ["{} {}".format(si, fi) for si,fi in zip(seqs, f)]
    out += ['']*(21-len(out))
    out = "\n".join(out) + '\n'
    
    with lock:
        print("\rfinished npos {:3d},  set {:3d}      ".format(npos, i), end='')
        with open("top20_{}.db".format(name), "at") as f:
            f.write(out)

# use a global var here for name, msa, weights so they do not need to be
# serialized by the processing pool - we can just access them in job.
# This way, when "fork"ing on unix systems, we can save memory/cache
# by page re-use from copy-on-write.
makedb_globals = ()
def makedb_job(arg):
    """
    choose a random set-of-positions, compute high-marg for those positions
    in all datasets, and call user-defined processing function with the marg
    """
    npos, i, seed = arg
    name, msa, weights, locks = makedb_globals
    rng = np.random.default_rng(seed)
    
    L = msa.shape[1]
    pos = np.sort(rng.choice(L, npos, replace=False))

    subseqs = np.ascontiguousarray(get_subseq(msa, pos))

    counts, uniq = highmarg([subseqs], weights=weights, return_uniq=True)
    f = counts[:,0]/np.sum(counts)
    #assert(N == (np.sum(weights) if weights is not None else msa.shape[0]))
    
    process_marg(name, npos, i, pos, f, uniq, locks[npos])

def make_db(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('reps', type=int)
    parser.add_argument('msa')
    parser.add_argument('--weights')
    parser.add_argument('--npos', default='2-10')
    args = parser.parse_args(args)

    reps = args.reps
    name = args.name

    dataseqs = seqload.loadSeqs(args.msa)[0]
    weights = None
    if args.weights is not None:
        weights = np.load(args.weights).astype('f4')
        print("using weights")

    Nd, L = dataseqs.shape
    msa = np.asfortranarray(dataseqs)
    
    npos_lo, npos_hi = [int(x) for x in args.npos.split('-')]
    npos_range = range(npos_lo, npos_hi+1)

    root_seed = np.random.SeedSequence()
    locks = {n: Lock() for n in npos_range}

    global makedb_globals
    makedb_globals = (name, msa, weights, locks)

    print("Starting workers...")
    print("")
    with Pool(os.cpu_count()) as pool:
        pool.map(makedb_job, ((n, i, root_seed.spawn(1)[0])
                              for n in npos_range for i in range(reps)))

###############################################################################
# Functions to search db and compute r20

count_msas_globals = ()
def count_job(arg):
    msas, weights = count_msas_globals
    pos, ref_seqs, fs = arg

    subseqs = [np.ascontiguousarray(get_subseq(m, pos)) for m in msas]
    dfs = countref(ref_seqs, subseqs, weights)

    if weights is not None:
        Ns = [m.shape[0] if w is None else np.sum(w)
              for m,w in zip(msas, weights)]
    else:
        Ns = [m.shape[0] for m in msas]
    dfs = dfs/np.array(Ns)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [pearsonr(fs, r)[0] for r in dfs.T]

def count_msas(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_name')
    parser.add_argument('msas', nargs='*')
    parser.add_argument('--weights', nargs='*')
    args = parser.parse_args(args)

    msas = [np.asfortranarray(seqload.loadSeqs(m)[0]) for m in args.msas]
    weights = None
    if args.weights:
        weights = [np.load(w) if w != 'None' else None for m in args.weights]

    positionsets = {}
    with open('top20_{}.db'.format(args.db_name), "rt") as f:
        while True:
            lines = [f.readline() for n in range(21)]
            if lines[0] == '':
                break
            pos = [int(p) for p in lines[0].split()]
            slines = (l.split() for l in lines[1:] if not l.isspace())
            seqs, fs = zip(*((s, float(f)) for s,f in slines))
            seqs = [[alpha.index(c) for c in s] for s in seqs]
            seqs = np.array(seqs, dtype='u1')
            fs = np.array(fs, dtype='f4')
            
            npos = len(pos)
            if npos not in positionsets:
                positionsets[npos] = []
            positionsets[npos].append((pos, seqs, fs))
    npos = list(positionsets.keys())
    npos.sort()

    global count_msas_globals
    count_msas_globals = (msas, weights)

    print("Starting workers...")
    with Pool(os.cpu_count()) as pool:
        print("Using {} workers".format(os.cpu_count()))
        res = pool.map(count_job, (d for n in npos for d in positionsets[n]))
    res = np.array(res).reshape((len(npos), -1, len(msas)))
    np.save(args.out_name, res)

###############################################################################
# this is used to development only to time performance

#def profile_count(args):
#    parser = argparse.ArgumentParser()
#    parser.add_argument('name')
#    parser.add_argument('msas', nargs='*')
#    args = parser.parse_args(args)

#    msas = [np.asfortranarray(seqload.loadSeqs(m)[0]) for m in args.msas]
    
#    npos = 10
#    with open("top20_{}_{}".format(args.name, npos), "rt") as f:
#        lines = [f.readline() for n in range(21)]
#        pos = [int(p) for p in lines[0].split()]
#        seqs, fs = zip(*((s, float(f)) 
#                         for s,f in (l.split() for l in lines[1:])))
#        seqs = [[alpha.index(c) for c in s] for s in seqs]
#        seqs = np.array(seqs, dtype='u1')
#        fs = np.array(fs, dtype='f4')
            
#    import cProfile
#    arg = (pos, seqs, fs, msas, None)
#    cProfile.runctx('count_job(arg)', {'count_job': count_job}, {'arg': arg})

###############################################################################

def main():
    funcs = {'make_db': make_db,
             #'profile_count': profile_count,
             'count': count_msas,
             }

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('action', choices=funcs.keys())

    known_args, remaining_args = parser.parse_known_args(sys.argv[1:])
    funcs[known_args.action](remaining_args)


if __name__ == '__main__':
    main()
