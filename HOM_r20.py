#!/usr/bin/env python
import sys, os, argparse, warnings, signal
from multiprocessing import Pool, set_start_method, shared_memory
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from numpy import random
import seqload
from scipy.stats import pearsonr, spearmanr
from highmarg import highmarg, countref

class SharedMSA:
    def __init__(self, msa, weights, shape=None):
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

def process_marg(topmode, npos, i, pos, f, uniq):
    if topmode == 'nonzero':
        top = f != 0
    if topmode.startswith('>'):
        top = f > float(topmode[1:])
    else:
        top = np.argsort(f)[-int(topmode):]

    seqs = ["".join(alpha[c] for c in si) for si in uniq[top]]
    f = f[top]

    out = [" ".join(str(p) for p in pos)]
    out += ["{} {}".format(si, fi) for si,fi in zip(seqs, f)]
    out = "\n".join(out) + '\n\n'

    return out

# use a global var here for msa, weights so they do not need to be
# serialized by the processing pool - we can just access them in job.
# This way, when "fork"ing on unix systems, we can save memory/cache
# by page re-use from copy-on-write.
def makedb_job(arg):
    """
    choose a random set-of-positions, compute high-marg for those positions
    in all datasets, and call user-defined processing function with the marg
    """
    try:
        npos, i, seed, topmode, shm_info = arg
        rng = np.random.default_rng(seed)
        
        shm = SharedMSA(*shm_info)

        L = shm.msa.shape[1]
        pos = np.sort(rng.choice(L, npos, replace=False))

        subseqs = np.ascontiguousarray(get_subseq(shm.msa, pos))

        #print("START", npos, i, repr(pos))
        counts, uniq = highmarg([subseqs], weights=shm.weights, return_uniq=True)
        #print("DONE", npos, i)
        freq = counts[:,0]/np.sum(counts)
        #assert(N == (np.sum(weights) if weights is not None else msa.shape[0]))

        shm.close()
        return npos, i, process_marg(topmode, npos, i, pos, freq, uniq)
    except Exception as e:
        print("ERROR", e)

def run_make_db(root_seed, topmode, npos_range, shm, reps, name):
    jobs = list( (n, i, root_seed.spawn(1)[0], topmode, shm.info())
                for n in npos_range[::-1] for i in range(reps))

    res = []
    remaining = set((n,i) for n,i,_,_,_ in jobs)

    print(f"Starting {os.cpu_count()} workers...")
    print(len(jobs))
    set_start_method('spawn')
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        #for n, i, dat in pool.imap_unordered(makedb_job, jobs):
        for n, i, dat in pool.map(makedb_job, jobs, chunksize=64):
            res.append(dat)
            remaining.difference_update([(n,i)])
            #print(f"\r{n: 3d} {i: 6d}  {len(remaining): 8d}       ", end="")
            print(f"{n: 3d} {i: 6d}  {len(remaining): 8d}   {max(remaining)}  {min(remaining)}       ")
            #if len(remaining) < 50:
            #    #print("")
            #    print(remaining, max(remaining), min(remaining))

    with open("{}.db".format(name), "wt") as f:
        f.write("".join(res))

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
    name = args.name.strip('.db')
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
        run_make_db(root_seed, topmode, npos_range, shm, reps, name)
    finally:
        shm.close()
        shm.unlink()

    print("Done!")

###############################################################################

from itertools import combinations
from more_itertools import set_partitions

def connected_corr(subseqs, s):
    N, L = subseqs.shape
    ids = subseqs == s  # should be fortran ordered

    # first construct dict of higher-order marginals to all orders for these
    # positions using stack-based algo to minimize computational operations
    pos = [0]
    idstack = [True, ids[:,0]]
    f = {}
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

    # DEBUG CODE/sanity check
    #pos = [0]
    #idstack = np.empty((L, N), dtype=bool)
    #idstack[0] = ids[:,0]
    #top = 0
    #f = {}
    #while True:
    #    #f[tuple(pos)] = np.mean(idstack[top])
    #    f[tuple(pos)] = np.sum(idstack[top])/float(N)
        
    #    if pos[-1] == L-1:
    #        pos.pop()
    #        if pos == []:
    #            break
    #        pos[-1] += 1
    #        if len(pos) == 1:
    #            idstack[0] = ids[:,pos[-1]]
    #        else:
    #            np.logical_and(ids[:,pos[-1]], idstack[top-2], out=idstack[top-1])
    #        top -= 1
    #    else:
    #        pos.append(pos[-1]+1)
    #        np.logical_and(ids[:,pos[-1]], idstack[top], out=idstack[top+1])
    #        top += 1
    # sanity check:
    #print(forig, f[tuple(range(L))])


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

def convert_cc_job(arg):
    (pos, seqs), shm_info = arg
    shm = SharedMSA(*shm_info)

    subseqs = get_subseq(shm.msa, pos)
    gs = [connected_corr(subseqs, si) for si in seqs]

    shm.close()
    return gs

def run_convert_cc_db(npos, positionsets, out_db_name, shm_info):

    jobs = [(d, shm_info) for n in npos for d in positionsets[n]]

    print(f"Starting {os.cpu_count()} workers...")
    set_start_method('spawn')
    with Pool(os.cpu_count()) as pool:
        res = pool.map(convert_cc_job, jobs)

    with open("{}.db".format(out_db_name), "wt") as f:
        for (pos, seqs), gs in zip(jobs, res):
            out = [" ".join(str(p) for p in pos)]
            seqs = ["".join(alpha[c] for c in si) for si in seqs]
            out += ["{} {}".format(si, fi) for si,fi in zip(seqs, gs)]
            out = "\n".join(out) + '\n\n'
            f.write(out)

def convert_cc_db(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_db_name')
    parser.add_argument('msa')
    parser.add_argument('--cutoff', type=int, default=20)
    args = parser.parse_args(args)

    out_db_name = args.out_db_name.strip('.db')
    db_name = args.db_name.strip('.db')

    msa = seqload.loadSeqs(args.msa)[0]

    positionsets = {}
    with open('{}.db'.format(db_name), "rt") as f:
        while True:
            lines = []
            while lines == [] or lines[-1] not in ['\n', '']:
                lines.append(f.readline())
            if lines[-1] == '':
                break

            pos = [int(p) for p in lines[0].split()]
            if len(pos) > args.cutoff:
                continue

            slines = (l.split() for l in lines[1:] if not l.isspace())
            seqs, fs = zip(*((s, float(f)) for s,f in slines))
            seqs = [[alpha.index(c) for c in s] for s in seqs]
            seqs = np.array(seqs, dtype='u1')
            fs = np.array(fs, dtype='f4')

            npos = len(pos)
            if npos not in positionsets:
                positionsets[npos] = []
            positionsets[npos].append((pos, seqs))
    npos = list(positionsets.keys())
    npos.sort(reverse=True)

    shm = SharedMSA(msa, None)

    try:
        run_convert_cc_db(npos, positionsets, out_db_name, shm.info())
    finally:
        shm.close()
        shm.unlink()

###############################################################################
# Functions to search db and compute r20

def count_job(arg):
    (pos, ref_seqs, fs), score, shm_info = arg

    shm = SharedMSA(*shm_info)
    msa, weights = shm.msa, shm.weights

    subseqs = np.ascontiguousarray(get_subseq(msa, pos))
    dfs = countref(ref_seqs, [subseqs], weights)

    if weights is not None:
        Ns = msa.shape[0] if weights is None else np.sum(weights)
    else:
        Ns = msa.shape[0]

    shm.close()

    dfs = dfs/Ns
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if score == 'pearson':
            return pearsonr(fs, r)[0]
        elif score == 'pearsonlog':
            return pearsonr(np.log(fs), np.log(r))[0]
        elif score == 'spearman':
            return spearmanr(fs, r)[0]
        elif score == 'pcttvd':
            s = 2*np.sum(fs)
            return np.sum(np.abs(r - fs))/s
        else:
            raise ValueError("invalid scoring method")

def run_count_msas(npos, positionsets, out_name, score, shm_info):
    jobs = ((d, score, shm_info) for n in npos for d in positionsets[n])

    print(f"Starting {os.cpu_count()} workers...")
    set_start_method('spawn')
    with Pool(os.cpu_count()) as pool:
        res = pool.map(count_job, jobs)

    res = np.array(res).reshape((len(npos), -1))
    np.save(out_name, res)

def count_msas(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_name')
    parser.add_argument('msa')
    parser.add_argument('--weights', nargs='*')
    parser.add_argument('--score', default='pearson',
                        choices=['pearson', 'pearsonlog', 'spearman', 'pcttvd'])
    args = parser.parse_args(args)

    msa = seqload.loadSeqs(m)[0]
    weights = None
    if args.weights:
        weights = [np.load(w) if w != 'None' else None for m in args.weights]

    positionsets = {}
    with open('{}.db'.format(args.db_name.rstrip('.db')), "rt") as f:
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
    npos = list(positionsets.keys())
    npos.sort()

    shm = SharedMSA(msa, weights)

    try:
        run_count_msas(npos, positionsets, args.out_name,
                       args.score, shm.info())
    finally:
        shm.close()
        shm.unlink()

###############################################################################
# Functions to search db and compute r20

def count_cc_job(arg):
    (pos, seqs, ogs), score, shm_info = arg
    print(pos)

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

def run_count_cc_msas(npos, positionsets, out_name, score, shm_info):
    jobs = ((d, score, shm_info) for n in npos for d in positionsets[n])

    print(f"Starting {os.cpu_count()} workers...")
    set_start_method('spawn')
    with Pool(os.cpu_count()) as pool:
        res = pool.map(count_cc_job, jobs)
    res = np.array(res).reshape((len(npos), -1))
    np.save(out_name, res)

def count_cc_msas(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('db_name')
    parser.add_argument('out_name')
    parser.add_argument('msa')
    parser.add_argument('--score', default='pearson',
                        choices=['pearson', 'spearman'])
    args = parser.parse_args(args)

    msa = seqload.loadSeqs(args.msa)[0]

    positionsets = {}
    with open('{}.db'.format(args.db_name.rstrip('.db')), "rt") as f:
        while True:
            lines = []
            while lines == [] or lines[-1] not in ['\n', '']:
                lines.append(f.readline())
            if lines[-1] == '':
                break

            pos = [int(p) for p in lines[0].split()]
            slines = (l.split() for l in lines[1:] if not l.isspace())
            seqs, gs = zip(*((s, float(f)) for s,f in slines))
            seqs = [[alpha.index(c) for c in s] for s in seqs]
            seqs = np.array(seqs, dtype='u1')
            gs = np.array(gs, dtype='f4')

            npos = len(pos)
            if npos not in positionsets:
                positionsets[npos] = []
            positionsets[npos].append((pos, seqs, gs))
    npos = list(positionsets.keys())
    npos.sort()

    shm = SharedMSA(msa, None)

    try:
        run_count_cc_msas(npos, positionsets, args.out_name,
                          args.score, shm.info())
    finally:
        shm.close()
        shm.unlink()

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
             'convert_cc_db': convert_cc_db,
             'count_cc': count_cc_msas,
             }

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('action', choices=funcs.keys())

    known_args, remaining_args = parser.parse_known_args(sys.argv[1:])
    funcs[known_args.action](remaining_args)


if __name__ == '__main__':
    main()
