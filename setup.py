from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyqsp',
    version='0.2.0',
    author='I. Chuang, J. Docter, J.M. Martyn, Z. Rossi, A. Tan',
    author_email='ichuang@mit.edu',
    packages=find_packages(include=['pyqsp', 'pyqsp.*']),
    url='https://github.com/ichuang/pyqsp',
    license='MIT',
    license_files=['LICENSE'],
    description='Generate phase angles for quantum signal processing algorithms.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pyqsp = pyqsp.main:CommandLine',
        ],
    },
    install_requires=['matplotlib',
                      'numpy>1.19.1',
                      'scipy',
                      ],
)
