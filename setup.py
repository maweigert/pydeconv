
from setuptools import setup, find_packages

setup(name='pydeconv',
    version='0.1.1',
    description='',
    url='',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=find_packages(),

    install_requires=["numpy","pyopencl","pyfftw"],

    package_data={"gputools":
                  ['convolve/kernels/*l',
                   'denoise/kernels/*',
                   'deconv/kernels/*',
                   'transforms/kernels/*',
                  ],
    },

    entry_points = {}
)
