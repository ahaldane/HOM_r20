all: highmarg.c art.c art.h seqtools.c
	python ./setup.py build_ext --inplace
