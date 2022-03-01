import glob

from setuptools import setup


def findfiles(pat):
    return [x for x in glob.glob('share/' + pat)]


data_files = [
]

with open("README.md", "r") as fh:
    long_description = fh.read()

# print "data_files = %s" % data_files

setup(
    name='pyqsp',
    version='0.1.6',
    author='I. Chuang, J. Docter, J.M. Martyn, Z. Rossi, A. Tan',
    author_email='ichuang@mit.edu',
    packages=['pyqsp', 'pyqsp.test', 'pyqsp.qsp_models'],
    scripts=[],
    url='https://github.com/ichuang/pyqsp',
    license='LICENSE.txt',
    description='Generate phase angles for quantum signal processing algorithms',
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
    package_dir={'pyqsp': 'pyqsp'},
    test_suite="pyqsp.test",
)
