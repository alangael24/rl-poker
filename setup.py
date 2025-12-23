from setuptools import setup, Extension
import numpy as np

poker_module = Extension(
    'poker_c',
    sources=['csrc/poker_binding.c'],
    include_dirs=[np.get_include(), 'csrc'],
    extra_compile_args=['-O3', '-march=native', '-ffast-math'],
)

setup(
    name='poker_c',
    version='1.0',
    description='High-performance Texas Hold\'em Poker in C',
    ext_modules=[poker_module],
)
