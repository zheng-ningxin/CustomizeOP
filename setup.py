import os
import setuptools
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

version = "0.1.1"

ext_modules = []

# Add the extral module of mkl sparse
mkl_sparse_ext = CppExtension(name='mkl_sparse_linear_cpp', sources=[
                              'mkl_sparse/mkl_spmm.cpp', 'mkl_sparse/mkl_sparse_conv.cpp'],
                              extra_compile_args=['-lmkl_core', '-lmkl_rt', '-fopenmp', '-Imkl_sparse', '-O3', '-D_GLIBCXX_USE_CXX11_ABI=0'])

omp_sparse_ext = CppExtension(name='omp_sparse_conv_cpp', sources=[
                              'omp_sparse/omp_sparse_conv.cpp'],
                              extra_compile_args=['-fopenmp', '-O3'])

ext_modules.append(mkl_sparse_ext)
ext_modules.append(omp_sparse_ext)

setup(
    name='CustomizedOp',
    version=version,
    description='The customized sparse OP for pytorch',
    author='MSRA',
    author_email="Ningxin.Zheng@microsoft.com",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
