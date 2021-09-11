#!/usr/bin/env python
import sys, os, argparse, warnings, signal
# use ProcessPoolExecutor instead of Pool because of better error handling
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
import numpy as np
from numpy import random
import seqload
from scipy.stats import pearsonr, spearmanr
from highmarg import highmarg, countref

# from python3.9
def removesuffix(self: str, suffix: str, /) -> str:
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]

# class to store MSA+weights in shared memory, to share between processes
# (since MSA is huge)
class SharedMSA:
    def __init__(self, msa, weights, shape=None):
        # user either provides MSAs to create shared mem for,
        # or provides the shared-mem lookup info, and we load it.
        if shape is None:
            self.create_shm(msa, weights)
        else:
            self.load_shm(msa, weights, shape)

    def info(self):
        w = None if self.weights is None else self.wshm.name
        return (self.sshm.name, w, self.msa.shape)

    def create_shm(self, msa, weights):
        # convert to shared memory in F-order
        self.sshm = shared_memory.SharedMemory(create=True, size=msa.nbytes)
        self.msa = np.ndarray(msa.shape, dtype='u1',
                              order='F', buffer=self.sshm.buf)
        self.msa[...] = msa[...]

        self.weights = None
        if weights is not None:
            self.wshm = shared_memory.SharedMemory(create=True, size=weights.nbytes)
            self.weights = np.ndarray(weights.shape, dtype='f4', buffer=self.wshm)
            self.weights[...] = weights[...]

        signal.signal(signal.SIGTERM | signal.SIGINT, self.handle_signal)

    def load_shm(self, msa, weights, shape):
        self.sshm = shared_memory.SharedMemory(name=msa)
        self.msa = np.ndarray(shape, dtype='u1', order='F', buffer=self.sshm.buf)

        self.weights = None
        if weights is not None:
            self.wshm = shared_memory.SharedMemory(name=weights)
            self.weights = np.ndarray(shape[0], dtype='f4', buffer=self.wshm.buf)

    def close(self):
        self.sshm.close()
        if self.weights is not None:
            self.wshm.close()

    def unlink(self):
        self.sshm.unlink()
        if self.weights is not None:
            self.wshm.unlink()

    def handle_signal(self, signum, frame):
        print("exiting due to signal " + str(signum))
        self.close()
        self.unlink()
        sys.exit(1)

# like executor.map, but prints out progress to stdout
def map_progress(executor, func, jobs):
    fs = [executor.submit(func, *j) for j in jobs]
    N = len(fs)

    # Yield must be hidden in closure so that the futures are submitted
    # before the first iterator value is required.
    def result_iterator():
        try:
            # reverse to keep finishing order
            fs.reverse()
            while fs:
                # Careful not to keep a reference to the popped future
                print(f"\rjobs remaining:  {len(fs): 10d}/{N}       ", end="")
                yield fs.pop().result()
        finally:
            for future in fs:
                future.cancel()
        print("")
    return result_iterator()


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

def read_db(db_name):
    positionsets = {}
    with open('{}.db'.format(db_name), "rt") as f:
        while True:
            lines = []
            while lines == [] or lines[-1] not in ['\n', '']:
                lines.append(f.readline())
            if lines[-1] == '':
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
    return positionsets

###############################################################################
# functions to generate db

def process_marg(topmode, npos, i, pos, f, uniq):
    if topmode == 'nonzero':
        top = f != 0
    if topmode.startswith('>'):
        top = f > float(topmode[1:])
    else:
        ntop = int(topmode)
        top = np.argpartition(f, -ntop)[-ntop:]
        top = top[np.argsort(f[top])]

    seqs = ["".join(alpha[c] for c in si) for si in uniq[top]]
    f = f[top]

    out = [" ".join(str(p) for p in pos)]
    out += ["{} {}".format(si, fi) for si,fi in zip(seqs, f)]
    out = "\n".join(out) + '\n\n'

    return out

def makedb_job(npos, i, seed, topmode, shm_info):
    """
    choose a random set-of-positions, compute high-marg for those positions
    in all datasets, and call user-defined processing function with the marg
    """
    #npos, i, seed, topmode, shm_info = arg
    rng = np.random.default_rng(seed)
    
    shm = SharedMSA(*shm_info)

    L = shm.msa.shape[1]
    pos = np.sort(rng.choice(L, npos, replace=False))

    subseqs = np.ascontiguousarray(get_subseq(shm.msa, pos))

    counts, uniq = highmarg([subseqs], weights=shm.weights, return_uniq=True)
    freq = counts[:,0]/np.sum(counts)
    #assert(N == (np.sum(weights) if weights is not None else msa.shape[0]))

    shm.close()

    return npos, i, process_marg(topmode, npos, i, pos, freq, uniq)

def make_db(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('reps', type=int)
    parser.add_argument('msa')
    parser.add_argument('--weights')
    parser.add_argument('--npos', default='2-10')
    parser.add_argument('--topmode', default='20',
                       help='integer, string of form ">0.01", or "nonzero"')
    args = parser.parse_args(args)

    reps = args.reps
    name = removesuffix(args.name, '.db')
    topmode = args.topmode

    dataseqs = seqload.loadSeqs(args.msa)[0]
    Nd, L = dataseqs.shape

    weights = None
    if args.weights is not None:
        weights = np.load(args.weights).astype('f4')
        print("using weights")

    npos_lo, npos_hi = [int(x) for x in args.npos.split('-')]
    npos_range = range(npos_lo, npos_hi+1)

    root_seed = np.random.SeedSequence()

    shm = SharedMSA(dataseqs, weights)
    try:
        jobs = list( (n, i, root_seed.spawn(1)[0], topmode, shm.info())
                    for n in npos_range[::-1] for i in range(reps))
        print(f"Starting {os.cpu_count()} workers...")
        res = []
        with ProcessPoolExecutor(os.cpu_count()) as executor:
            futures = [executor.submit(makedb_job, *j) for j in jobs]
            for future in as_completed(futures):
                n, i, dat = future.result()
                res.append(dat)
                print(f"\r{n: 3d} {i: 6d}    ", end="")
    finally:
        shm.close()
        shm.unlink()

    with open("{}.db".format(name), "wt") as f:
        f.write("".join(res))

    print("Done!")

###############################################################################

from itertools import combinations
from more_itertools import set_partitions

def connected_corr(subseqs, s):
    N, L = subseqs.shape

    # for each column, keep track of matches (bool seq-ident)
    ids = subseqs == s  # should be fortran ordered

    # first construct dict of higher-order marginals to all orders for these
    # positions using stack-based algo to minimize computational operations
    pos = [0]  # current set-of-positions
    idstack = [True, ids[:,0]] # stack of total seq match bools
    f = {}  # dict of computed marginals
    while True:
        f[tuple(pos)] = np.mean(idstack[-1])
        
        if pos[-1] == L-1:
            pos.pop()
            idstack.pop()
            if pos == []:
                break
            pos[-1] += 1
            idstack[-1] = ids[:,pos[-1]] & idstack[-2]
        else:
            pos.append(pos[-1]+1)
            idstack.append(ids[:,pos[-1]] & idstack[-1])

    # now use the formula:
    # g_1..n = f_1..n - \sum_P \prod_Pi g_Pi
    # where P are all the possible partitions of 1..n into unique sets
    # and Pi is each part of the partition

    # the n=1 g terms are just the unimarg
    gs = {(p,): f[(p,)] for p in range(L)}

    # next compute g orders n=2 to L
    for n in range(2, L+1):  # n = order of g to compute
        for pos in combinations(range(L), n):  # subsequence positions to consider
            # sum over g-partition products
            gsum = 0
            for l in range(2, n+1):  # l = number of partion groups
                for p in set_partitions(pos, l):
                    gsum += np.product([gs[tuple(pi)] for pi in p])

            gs[pos] = f[pos] - gsum

    return gs[pos]

def convert_cc_job(pos, seqs, shm_info):
    shm = SharedMSA(*shm_info)

    subseqs = get_subseq(shm.msa, pos)
    gs = [connected_corr(subseqs, si) for si in seqs]

    shm.close()
    return gs

def convert_cc_db(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_db_name')
    parser.add_argument('msa')
    parser.add_argument('--cutoff', type=int, default=20)
    args = parser.parse_args(args)

    out_db_name = removesuffix(args.out_db_name, '.db')
    db_name = removesuffix(args.db_name, '.db')

    msa = seqload.loadSeqs(args.msa)[0]

    positionsets = read_db(db_name)
    positionsets = {n: d for n,d in positionsets.items() 
                    if n <= args.cutoff}

    npos = list(positionsets.keys())
    npos.sort(reverse=True)

    shm = SharedMSA(msa, None)
    try:
        jobs = [(pos, seqs, shm.info()) for n in npos
                for pos, seqs, fs in positionsets[n]]
        print(f"Starting {os.cpu_count()} workers...")
        with ProcessPoolExecutor(os.cpu_count()) as executor:
            res = list(map_progress(executor, convert_cc_job, jobs))
    finally:
        shm.close()
        shm.unlink()

    with open("{}.db".format(out_db_name), "wt") as f:
        for (pos, seqs, _), gs in zip(jobs, res):
            out = [" ".join(str(p) for p in pos)]
            seqs = ["".join(alpha[c] for c in si) for si in seqs]
            out += ["{} {}".format(si, fi) for si,fi in zip(seqs, gs)]
            out = "\n".join(out) + '\n\n'
            f.write(out)

###############################################################################
# Functions to search db and compute r20

def count_job(pos, ref_seqs, fs, score, shm_info):
    shm = SharedMSA(*shm_info)
    msa, weights = shm.msa, shm.weights

    subseqs = np.ascontiguousarray(get_subseq(msa, pos))
    dfs = countref(ref_seqs, [subseqs], weights)[:,0]

    if weights is not None:
        Ns = msa.shape[0] if weights is None else np.sum(weights)
    else:
        Ns = msa.shape[0]

    shm.close()

    dfs = dfs/Ns
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if score == 'pearson':
            return pearsonr(dfs, fs)[0]
        elif score == 'pearsonlog':
            return pearsonr(np.log(dfs), np.log(fs))[0]
        elif score == 'spearman':
            return spearmanr(dfs, fs)[0]
        elif score == 'pcttvd':
            s = 2*np.sum(dfs)
            return np.sum(np.abs(dfs - fs))/s
        else:
            raise ValueError("invalid scoring method")

def count_msas(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_name')
    parser.add_argument('msa')
    parser.add_argument('--weights')
    parser.add_argument('--score', default='pearson',
                        choices=['pearson', 'pearsonlog', 'spearman', 'pcttvd'])
    args = parser.parse_args(args)

    msa = seqload.loadSeqs(args.msa)[0]
    weights = None
    if args.weights:
        weights = np.load(w).astype('f4')


    db_name = removesuffix(args.db_name, '.db')
    positionsets = read_db(db_name)
    npos = list(positionsets.keys())
    npos.sort(reverse=True) # process from largest to smallest

    shm = SharedMSA(msa, weights)
    try:
        jobs = ((pos, seqs, fs, args.score, shm.info()) for n in npos
                 for pos, seqs, fs in positionsets[n])
        print(f"Starting {os.cpu_count()} workers...")
        with ProcessPoolExecutor(os.cpu_count()) as executor:
            res = list(map_progress(executor, count_job, jobs))
    finally:
        shm.close()
        shm.unlink()

    res = np.array(res).reshape((len(npos), -1))[::-1,:] # smallest first
    np.save(args.out_name, res)

###############################################################################
# this creates a new db from an existing one by computing the same marginals
# for another MSA

def recompute_db_job(pos, seqs, shm_info):
    shm = SharedMSA(*shm_info)
    msa, weights = shm.msa, shm.weights

    subseqs = np.ascontiguousarray(get_subseq(msa, pos))
    dfs = countref(seqs, [subseqs], weights)[:,0]

    shm.close()

    Ns = msa.shape[0] if weights is None else np.sum(weights)
    return dfs/Ns

def recompute_db_msas(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_name')
    parser.add_argument('msa')
    parser.add_argument('--weights')
    args = parser.parse_args(args)

    msa = seqload.loadSeqs(args.msa)[0]
    weights = None
    if args.weights:
        weights = np.load(w).astype('f4')

    out_name = removesuffix(args.out_name, '.db')
    db_name = removesuffix(args.db_name, '.db')
    positionsets = read_db(db_name)
    npos = list(positionsets.keys())
    npos.sort(reverse=True) # process from largest to smallest

    shm = SharedMSA(msa, weights)
    try:
        jobs = [(pos, seqs, shm.info()) for n in npos
                 for pos, seqs, fs in positionsets[n]]
        print(f"Starting {os.cpu_count()} workers...")
        with ProcessPoolExecutor(os.cpu_count()) as executor:
            res = list(map_progress(executor, recompute_db_job, jobs))
    finally:
        shm.close()
        shm.unlink()

    with open("{}.db".format(out_name), "wt") as f:
        for (pos, seqs, _), fs in zip(jobs, res):
            out = [" ".join(str(p) for p in pos)]
            seqs = ["".join(alpha[c] for c in si) for si in seqs]
            out += ["{} {}".format(si, fi) for si,fi in zip(seqs, fs)]
            out = "\n".join(out) + '\n\n'
            f.write(out)

###############################################################################
# Functions to search db and compute r20

def count_cc_job(pos, seqs, ogs, score, shm_info):
    shm = SharedMSA(*shm_info)
    msa, weights = shm.msa, shm.weights

    subseqs = get_subseq(msa, pos)
    gs = [connected_corr(subseqs, si) for si in seqs]

    shm.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if score == 'pearson':
            return pearsonr(ogs, gs)[0]
        elif score == 'spearman':
            return spearmanr(ogs, gs)[0]
        else:
            raise ValueError("invalid scoring method")

def count_cc_msas(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_name')
    parser.add_argument('msa')
    parser.add_argument('--score', default='pearson',
                        choices=['pearson', 'spearman'])
    args = parser.parse_args(args)

    msa = seqload.loadSeqs(args.msa)[0]

    db_name = removesuffix(args.db_name, '.db')
    positionsets = read_db(db_name)
    npos = list(positionsets.keys())
    npos.sort(reverse=True) # process from largest to smallest

    shm = SharedMSA(msa, None)
    try:
        jobs = ((pos, seqs, gs, args.score, shm.info()) for n in npos
                for pos, seqs, gs in positionsets[n])
        print(f"Starting {os.cpu_count()} workers...")
        with ProcessPoolExecutor(os.cpu_count()) as executor:
            res = list(map_progress(executor, count_cc_job, jobs))
    finally:
        shm.close()
        shm.unlink()

    res = np.array(res).reshape((len(npos), -1))[::-1,:] # smallest first
    np.save(args.out_name, res)

###############################################################################
# this is used for development only to time performance

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
             'recompute_db': recompute_db_msas,
             'convert_cc_db': convert_cc_db,
             'count_cc': count_cc_msas,
             }

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('action', choices=funcs.keys())

    known_args, remaining_args = parser.parse_known_args(sys.argv[1:])
    funcs[known_args.action](remaining_args)


if __name__ == '__main__':
    main()
