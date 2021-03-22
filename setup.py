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
    version='0.0.2',
    author='I. Chuang',
    author_email='ichuang@mit.edu',
    packages=['pyqsp', 'pyqsp.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/pyqsp/',
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
    install_requires=['numpy',
                      'scipy',
                      ],
    package_dir={'pyqsp': 'pyqsp'},
    test_suite="pyqsp.test",
)
