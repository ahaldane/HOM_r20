#!/usr/bin/env python
from numpy.distutils.core import setup, Extension

# compile me with
# python ./setup.py build_ext --inplace

module = Extension('highmarg',
                    sources = ['highmarg.c', 'art.c'],
                    extra_compile_args = ['-O3', '-Wall'])
setup (name = 'highmarg',
       version = '1.0',
       description = 'higher order marginal calculator',
       ext_modules = [module])


module = Extension('seqtools',
                    sources = ['seqtools.c'],
                    extra_compile_args = ['-O3', '-Wall'])
setup (name = 'seqtools',
       version = '1.0',
       description = 'helper functions for sequence analysis',
       ext_modules = [module])
