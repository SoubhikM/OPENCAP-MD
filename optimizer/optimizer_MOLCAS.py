#!/usr/bin/env python

'''Copyright (c) 2024 Soubhik Mondal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
"""
#####################################################################################
#                                                                                   #
#            Geometry Optimization Algorithms Integrated in This Code:              #
#                                                                                   #
#  - Berny:     https://github.com/jhrmnn/pyberny                                   #
#  - geomeTRIC: https://github.com/leeping/geomeTRIC                                #
#  - OptKing:   https://github.com/psi-rking/optking                                #
#                                                                                   #
#####################################################################################
"""

'''
Utilizes SHARC-OPENCAP interface (MOLCAS).
Author: Soubhik Mondal,
Boston University.
'''

import numpy as np
import sys, os
import subprocess as sp
import time
from tabulate import tabulate
import geometric
import geometric.molecule
from berny import Berny, geomlib
from berny import optimize
import re
import logging
import datetime
import copy
#sys.path.append('./../utils/')

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
utils_dir = os.path.join(script_dir, "../utils")
sys.path.append(utils_dir)

EV_TO_AU = 1 / 27.21138602
ANG_TO_BOHR = 1.8897259886
AMU_TO_AU = 1. / 5.4857990943e-4


def init_params(geomfile):
    """
    Reads geometry file in SHARC/COLUMBUS format

    Parameters:
    ----------
    geomfile
        filename (geom)

    Returns:
    -------
        geometry: np.ndarray

        atom_symbols

    """

    geomstrings = readfile(geomfile)
    geom = []
    atom_symbols = []

    for i, geom_line in enumerate(geomstrings):
        if not geom_line.strip():
            continue
        geom.append([float(_geom) for _geom in geom_line.split()[2:5]])
        atom_symbols.append(geom_line.split()[0])
    geom = np.asarray(geom)
    atom_symbols = np.asarray(atom_symbols)

    return geom, atom_symbols


def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out


def readQMout(QMoutFileName, Natom):
    '''
    Read QM.out file and reads energy and gradient
    '''
    QMoutlines = readfile(QMoutFileName)
    gradQM = []
    HamQM = []
    for i, line in enumerate(QMoutlines):
        if " Hamiltonian Matrix" in line:
            HamQM.append(float(QMoutlines[i + 2].split()[0]))

        if "Gradient Vectors" in line:
            for _, gradline in enumerate(QMoutlines[i + 2:i + Natom + 2]):
                gradQM.append([float(_grad) for _grad in gradline.split()])
            break
    gradQM = np.asarray(gradQM)
    HamQM = np.asarray(HamQM)
    return gradQM, HamQM


def writefile(filename, content):
    '''
    content can be either a string or a list of strings
    '''
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(13)


def makeQMin(QMin):
    '''
    Creates the QM.in file in SHARC-MOLCAS format!
    '''
    QMinfilename = "%s/QM/QM.in" % QMin['pwd']
    string = ''
    string += f"{int(QMin['natom'])}\n"
    string += f'\tThis is a job run by: {sys.argv[0]}\n'

    for idx, (x, y, z) in enumerate(QMin['coords']):
        string += '%s\t%16.12f\t%16.12f\t%16.12f\t%13.12f\t%13.12f\t%13.12f\n' % (QMin['atom_symbols'][idx],
                                                                                  x / ANG_TO_BOHR, y / ANG_TO_BOHR,
                                                                                  z / ANG_TO_BOHR,
                                                                                  0.0, 0.0, 0.0)

    string += 'unit\tangstrom\n'
    string += 'states\t1\n'
    string += 'dt\t0.00000\n'
    string += 'step\t%i\n' % (QMin['iter'])
    string += "savedir\t%s\n" % (os.path.join(QMin['pwd'], "restart"))
    string += 'H\nDM\nGRAD all\nNACDR\nPCAP\n'
    if QMin['iter'] == 0:
        string += 'init\n'

    writefile(QMinfilename, string)
    return


def make_geomfile(QMin):
    '''Make geometry file in SHARC/COLUMBUS format'''
    geomflines = ''
    import isotope_mass
    for label, (x, y, z) in zip(QMin['atom_symbols'], QMin['coords']):
        atomic_weight = isotope_mass.MASSES.get(label)
        atomic_number = isotope_mass.get_atomic_number(label)
        geomflines += "%s\t%s\t%16.12f\t%16.12f\t%16.12f\t%16.12f\n" % (
        label, atomic_number, x, y, z, atomic_weight)

    writefile("%s/geom.restart" % QMin['pwd'], geomflines)
    return


def create_bash_script(script_path, fname, QMin):
    """
    Create runQM.sh bash file in QM directory in SHARC format to run SHARC_MOLCAS_OPENCAP.py interface
    :param script_path: path of SHARC_MOLCAS_OPENCAP.py file
    :param fname: file-name to print: QM/runQM.sh
    :param QMin:
    :return: QM/runQM.sh file
    """
    bash_script = '#!/bin/sh\nexport OMP_NUM_THREADS={}\n\n\n' \
                  'export SCRIPT_PATH={}\n' \
                  'chmod +rwx "$SCRIPT_PATH"\n\n' \
                  'cd QM || exit\n\n' \
                  'sed -i "s/SOC/H/g" QM.in\n' \
                  '"$SCRIPT_PATH" QM.in >> QM.log 2>> QM.err\n\n' \
                  'exit $?'.format(str(QMin['ncpu']), script_path)

    with open(fname, "w") as file:
        file.write(bash_script)

def run_calc(QMin):
    '''
    Run the SHARC-OPENMOLCAS interface using runQM.sh file in QM directory
    '''

    pathRUN = QMin['pwd']
    if not os.path.exists("%s/QM/runQM.sh" % pathRUN):
        interface_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "../sharc-interface/")
        create_bash_script(os.path.join(interface_dir, "SHARC_MOLCAS_OPENCAP.py"), f"{pathRUN}/QM/runQM.sh", QMin)

    start_time = time.time()
    stringrun = "sh " + "%s/QM/runQM.sh" % pathRUN
    pathRUN = "./"
    stdoutfile = open("%s/QM/QM.stdout" % pathRUN, 'w')
    stderrfile = open("%s/QM/QM.stderr" % pathRUN, 'w')

    try:
        runerror = sp.run(stringrun.split(), cwd=pathRUN, stdout=stdoutfile, stderr=stderrfile, shell=False).returncode
        pass
    except OSError as e:
        print(f'Call{stringrun}\n ended with error{e}')
        sys.exit(75)
    finally:
        stdoutfile.close()
        stderrfile.close()
    end_time = time.time()

    print_to_file("\n===> Time elapsed for electronic structure run: %10.8f s.\n" % (end_time - start_time), pathRUN)
    return runerror


def print_to_table(title, headers, data):
    """
    Print data to table
    :param title: (str) Title information
    :param headers: list(str) Column headers information
    :param data: list(zip(data)), all data in zipped and list format
    :return: A printed table
    """
    rows = np.shape(data)[0]
    string = []

    for irow in range(rows):
        rstring = []
        vals = [val for val in data[irow]]
        for _val in vals:
            if isinstance(_val, float):
                rstring.append("% 8.6f" % _val)
            elif isinstance(_val, complex):
                rstring.append("(% 8.6f, % 8.6f)" % (_val.real, _val.imag))
            elif isinstance(_val, int):
                rstring.append("% 2i" % (_val))
            else:
                rstring.append(str(_val))
        string.append(rstring)

    from tabulate import tabulate
    print(f"\n{title.center(10)}\n")
    #    table = tabulate(string, headers=headers, tablefmt='pretty', numalign="center", stralign="center")
    table = tabulate(string, headers=headers, tablefmt='pretty', numalign="center", stralign="center", floatfmt='g')
    print(table)


def print_string(info, elements, matrix):
    '''Print matrices for elements'''
    float_format = "{: 12.8f}"
    ostring = str(info) + "\n"
    headers = ['Atom', 'X', 'Y', 'Z']
    table_data = []
    for atom_index, atom_matrix in enumerate(matrix, start=1):
        formatted_geom = [float_format.format(_matrix) for _matrix in atom_matrix]
        table_data.append([elements[atom_index - 1]] + formatted_geom)
    ostring += tabulate(table_data, headers=headers, tablefmt="pretty", numalign="center", stralign="right",
                        floatfmt='g')
    return ostring


def print_to_file(ostring, pathRUN):
    """Prints to optimizer.molcas.log file"""
    with open("%s/optimizer.molcas.log" % pathRUN, "a+") as f:
        f.write(ostring + "\n")
    return


def SHARC_OPENCAP_run(QMin):
    """A wrapper around run_calc function"""
    makeQMin(QMin)
    runerror = run_calc(QMin)
    if runerror != 0:
        print("SEVERE error in MOLCAS run: Runerror %s: " % runerror)
        sys.exit(911)

    QMoutFileName = '%s/QM/QM.out' % QMin['pwd']
    gradients, e_tot = readQMout(QMoutFileName, QMin['natom'])

    return e_tot[0], gradients


# Optimization engines are defined now
class CustomEngine(geometric.engine.Engine):
    """
    A custom engine built for geometric optimizer
    """
    def __init__(self, molecule, QMin):
        self.mol = molecule
        self.rundir = QMin['pwd']
        super(CustomEngine, self).__init__(molecule)
        self._iter = 0
        if 'restart' in QMin:
            self._iter = QMin['iter']
        self.qmin = QMin

    def calc_new(self, coords, dirname):
        coords = coords.reshape(len(self.mol.elem), 3)
        coordsnew = [(e, xyz) for e, xyz in zip(self.mol.elem, coords)]

        self.qmin['coords'] = copy.deepcopy(coords)
        make_geomfile(self.qmin)  # For restart

        starttime = datetime.datetime.now()

        print_to_file("\n\nMOLCAS run: %i\n" % self._iter, self.rundir)
        print_to_file(print_string("Position", self.mol.elem, coords), self.rundir)

        energy, gradient = SHARC_OPENCAP_run(self.qmin)

        print_to_file("\nEnergy: %10.8f Hartree\n\n" % energy, self.rundir)
        print_to_file(print_string("Gradients (Hartree/Bohr)", self.mol.elem, gradient), self.rundir)

        endtime = datetime.datetime.now()
        print_to_file("\nElapsed time (hh:mm:ss) ==> %s\n\n" % (endtime - starttime), self.rundir)

        self._iter += 1
        self.qmin['iter'] = self._iter
        return {'energy': energy, 'gradient': gradient.ravel()}

# Berny Solver
def BernySolver(engine, QMin):
    _iter = 0
    if 'restart' in QMin:
        _iter = QMin['iter']

    _atoms, _lattice = yield
    pathRUN = str(QMin['pwd'])

    while True:
        coords = np.array([coord for _, coord in _atoms])
        elems = np.array([el for el, _ in _atoms])
        QMin['coords'] = copy.deepcopy(coords)
        make_geomfile(QMin)

        starttime = datetime.datetime.now()
        print_to_file("\n\nMOLCAS run: %i\n" % QMin['iter'], pathRUN)
        print_to_file(print_string("Position", elems, coords), pathRUN)

        energy, gradients = engine(QMin)

        print_to_file("\nEnergy: %10.8f Hartree\n\n" % energy, pathRUN)
        print_to_file(print_string("Gradients (Hartree/Bohr)", elems, gradients), pathRUN)

        endtime = datetime.datetime.now()
        print_to_file("\nElapsed time (hh:mm:ss) ==> %s\n\n" % (endtime - starttime), pathRUN)

        _iter += 1
        QMin['iter'] = _iter
        _atoms, _lattice = yield energy, gradients

class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', min_time_difference=10):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.previous_time = None
        self.min_time_difference = min_time_difference

    def format(self, record):
        current_time_seconds = record.created
        current_time_str = self.formatTime(record, self.datefmt)

        if self.previous_time and (current_time_seconds - self.previous_time) < self.min_time_difference:
            current_time_str = ''
        else:
            self.previous_time = current_time_seconds

        formatted_message = super().format(record)
        if current_time_str:
            return f"\n\n{'=' * 80}\n\n{current_time_str} - {record.name} - {record.levelname} - {formatted_message}"
        else:
            return f"{formatted_message}"

def optkingSolver(engine, QMin):
    '''Optking engine'''
    _iter = 0
    if 'restart' in QMin:
        _iter = QMin['iter']

    pathRUN = str(QMin['pwd'])

    import optking
    import qcelemental as qcel

    atomic_numbers = [qcel.periodictable.to_Z(symbol) for symbol in QMin['atom_symbols']]
    geometry = np.column_stack((atomic_numbers, QMin['coords'] / ANG_TO_BOHR))
    mol = qcel.models.Molecule.from_data(geometry, dtype="numpy")
    conv_type = QMin.get('conv') or "gau_tight"  # gau_tight is the default choice
    optking_options = {"g_convergence": conv_type,
                       "accept_symmetry_breaking": True,
                       "ADD_AUXILIARY_BONDS": True,
                       "INCLUDE_OOFP": True,
                       "opt_coordinates": "both",
                       "ENSURE_BT_CONVERGENCE": True}

    try:
        optking_options_extra = read_optking_params(QMin)
        optking_options.update(optking_options_extra)
    except OSError as e:
        print('Reading of optking.params returned:\t', e)

    opt = optking.CustomHelper(mol, optking_options)

    for step in range(_iter, 300):
        starttime = datetime.datetime.now()
        print_to_file("\n\nMOLCAS Solver run: %i\n" % QMin['iter'], pathRUN)
        print_to_file(print_string("Position", QMin['atom_symbols'], QMin['coords']), pathRUN)

        E, gX = engine(QMin)
        opt.E = E
        opt.gX = gX
        opt.compute()
        opt.take_step()
        conv = opt.test_convergence()

        # Update coords for the next run!
        QMin['coords'] = copy.deepcopy(opt.geom)
        make_geomfile(QMin)

        print_to_file("\nEnergy: %10.8f (Hartree)\n\n" % E, pathRUN)
        print_to_file(print_string("Gradients (Hartree/Bohr)", QMin['atom_symbols'], gX), pathRUN)

        endtime = datetime.datetime.now()
        print_to_file("\nElapsed time (hh:mm:ss) ==> %s\n\n" % (endtime - starttime), pathRUN)

        _iter += 1
        QMin['iter'] = _iter

        if conv is True:
            print("Optimization SUCCESS after %i iterations:" % (_iter + 1))
            break
    else:
        print("Optimization FAILURE (iteration: %i)\n" % (_iter + 1))

    return opt.geom

def read_optking_params(QMin):
    """
    Read optking.params file in PWD in case optking optimizer is used
    In absence default params are used

    Parameters:
    ----------
    QMin

    Returns:
    -------
        optking_extras: dict, extra optimizer parameters
    """
    optking_extras = {}
    fname = os.path.join(QMin['pwd'], 'optking.params')
    with open(fname, 'r') as f:
        for line in f:
            line = line.split("#")[0].strip()  # Remove comments
            if line:
                try:
                    key, value = line.split(None, 1)
                    key = key.strip()
                    value = value.strip()
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    optking_extras[key] = value
                except ValueError:
                    print(f"Line format error: {line} in optking.params file!")
    return optking_extras





# Read MOLCAS.template, resources file (written to then QM.in file)
def _build_QMin(QMin):
    '''Read MOLCAS.template, resources file (stored in QMin dict)'''

    # Open and read the template file
    template_path = os.path.join(QMin['pwd'], 'QM', 'MOLCAS.template')
    template = readfile(template_path)
    resources_path = os.path.join(QMin['pwd'], 'QM', 'MOLCAS.resources')
    resources = readfile(resources_path)
    string = "\n\n====> MOLCAS.resources/template variables for optimizer:\n\n"

    for line in template:
        line = re.sub('#.*$', '', line).strip().lower().split()

        if not line:
            continue

        key = line[0]
        value = line[1] if len(line) > 1 else None
        if key == 'act_states':
            QMin[key] = [int(value)]
        if key in QMin:
            string += f"\t{key}: {QMin[key]}\n"

    for line in resources:
        # Clean up the line and split it
        line = re.sub('#.*$', '', line).strip().split()
        if not line:
            continue  # Skip empty lines

        key = line[0].lower()
        value = line[1] if len(line) > 1 else None

        if key == 'restart':
            QMin['iter'] = int(value)
            QMin[key] = []
        elif key == 'optimizer':
            QMin[key] = str(value).lower()
            if QMin[key] == 'optking' or QMin[key] == 'geometric':
                if len(line) > 2:
                    QMin['conv'] = str(line[2]).lower()
        elif key == 'debug':
            QMin['debug'] = []
        elif key == 'gradientmax':
            QMin[key] = float(value)
        elif key == 'gradientrms':
            QMin[key] = float(value)
        elif key == 'stepmax':
            QMin[key] = float(value)
        elif key == 'steprms':
            QMin[key] = float(value)
        elif key == 'deltae':
            QMin[key] = float(value)
        elif key == 'superweakdih':
            QMin[key] = True
        elif key == 'trust':
            QMin[key] = float(value)
        if key in QMin:
            string += f"\t{key}: {QMin[key]}\n"

    string += "\n"
    print_to_file(string, QMin['pwd'])

    return QMin


def main():
    QMin = {}
    QMin['pwd'] = str(sys.argv[1])
    QMin['optimizer'] = 'geometric'
    QMin['iter'] = 0

    # Optimizer defaults
    QMin['gradientmax'] = 4.50E-04
    QMin['gradientrms'] = 3.00E-04
    QMin['stepmax'] = 1.80E-03
    QMin['steprms'] = 1.20E-03
    QMin['deltae'] = 1.00e-06
    QMin['superweakdih'] = False
    QMin['trust'] = 0.3

    QMin = _build_QMin(QMin)
    if 'restart' in QMin:
        geomfile = "%s/geom.restart" % QMin['pwd']
    else:
        geomfile = "%s/geom" % QMin['pwd']

    initial_positions, atom_symbols = init_params(geomfile)
    QMin['atom_symbols'] = atom_symbols
    QMin['natom'] = len(QMin['atom_symbols'])
    QMin['coords'] = initial_positions

    print_to_file("optimizer run path:\n %s" % QMin['pwd'], QMin['pwd'])


    # =============================
    if QMin.get('optimizer') == 'berny':
        logger = logging.getLogger('MOLCAS_pyBerny_logger')
        if 'debug' in QMin:
            logger.setLevel(logging.DEBUG)  # or any other level like DEBUG, WARNING, etc., DEBUG prints everything
        else:
            logger.setLevel(logging.INFO)

        formatter = CustomFormatter('%(message)s')
        handler = logging.FileHandler('berny.log')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        conv_params = {
            'gradientmax': QMin['gradientmax'],  # Eh/Bohr
            'gradientrms': QMin['gradientrms'],  # Eh/Bohr
            'stepmax': QMin['stepmax'],  # Bohr
            'steprms': QMin['steprms'],  # Bohr
            'trust': 0.3,
            'dihedral': True,
            'superweakdih': QMin['superweakdih']
        }

        from berny import berny
        berny.defaults = conv_params  # Overwrite some defaults!

        geom_init = geomlib.Geometry(species=atom_symbols, coords=initial_positions)
        optimizer = Berny(geom_init, logger=logger)

        # Optimize by pyberny
        solverBerny = BernySolver(SHARC_OPENCAP_run, QMin)
        geom_final = optimize(optimizer, solverBerny, trajectory=None)

        # Print the final geometry
        geom_final = np.array([coord for _, coord in geom_final])
        print(print_string("Optimized positions of the atoms:", atom_symbols, geom_final))

    elif QMin.get('optimizer') == 'geometric':
        # Default choice
        # Note to SBK: QM.in has information in Hartree and Bohr, TRIC needs initial geometry in Angstrom, then proceeds
        # To calculate parameters in atomic units, Hence we need to do unit conversion only once!
        molecule = geometric.molecule.Molecule()
        molecule.elem = [elem for elem in atom_symbols]
        molecule.xyzs = [initial_positions / ANG_TO_BOHR]

        conv_params = ['energy', QMin['deltae'],
                       'grms', QMin['gradientrms'],
                       'gmax', QMin['gradientmax'],
                       'drms', QMin['steprms'],
                       'dmax', QMin['stepmax']]

        outfile = os.path.join(QMin['pwd'], 'QM', "geomeTRIC.log")
        conv_type = QMin.get('conv') or "gau"
        with open(outfile, "w+") as _outfile:
            m = geometric.optimize.run_optimizer(customengine=CustomEngine(molecule, QMin),
                                                 converge=conv_params,
                                                 #convergence_set = conv_type,
                                                 check=1,
                                                 input=_outfile.name)

        print(print_string("Optimized positions of the atoms (Angstrom):", atom_symbols, m.xyzs[-1]))

    elif QMin.get('optimizer') == 'optking':
        geom_final = optkingSolver(SHARC_OPENCAP_run, QMin)
        print(print_string("Optimized positions of the atoms (Angstrom):", atom_symbols, geom_final))

if __name__ == "__main__":
    main()
