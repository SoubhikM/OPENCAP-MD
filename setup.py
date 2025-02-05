"""**OPENCAP-MD**: A package that calculates gradients and couplings on the fly for metastable states molecular dynamics

**OPENCAP-MD** is an open-source package aimed at extending the capabilities of bound state electronic structure packages to
describe metastable electronic states' dynamics by calculating gradients and couplings on-the-fly.

Currently supported electronic structure packages are:

- **OpenMolcas (CASSCF)** (https://gitlab.com/Molcas/OpenMolcas)
- **Columbus (SA-MCSCF,MR-CIS)** (https://gitlab.com/columbus-program-system/columbus)

Currently available features:

- Metastable state gradients and couplings in presence of a box CAP
- Natural orbitals of metastable state root (real part of complex 1-RDMs)

This project of OPENCAP-MD is distributed under multiple licenses:

- Some files (derived from SHARC) are licensed under **GNU General Public License v3.0 (GPLv3)**.
- Other files, written independently, are licensed under **MIT**.

See individual source files for specific licensing information.

"""

from setuptools import setup, find_packages

setup(
    name='opencap-md',
    version='0.1.0.dev',
    author='Soubhik Mondal',
    author_email='soubhikm@bu.edu',
    description='OPENCAP-MD: A package that calculates gradients and couplings on the fly for metastable states molecular dynamics.',
    project_urls={
        'Source code': "https://github.com/SoubhikM/OPENCAP-MD",
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'geometric',
        'pyopencap',
        'tabulate',
        'h5py',
        'pyberny',
        'mendeleev',
        'pyscf',
        'optking',
        'qcelemental'
    ],
    python_requires='>=3.7',
)

