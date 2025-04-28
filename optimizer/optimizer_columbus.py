#!/usr/bin/env python3
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

import numpy as np
import sys, os
import scipy.linalg as LA
import copy
import shutil
import pyopencap
from pyopencap.analysis import CAPHamiltonian
from pyopencap.analysis import colparser, colparser_mc
from tabulate import tabulate
import geometric
import geometric.molecule
import h5py
import subprocess as sp
import datetime, time
from functools import reduce
import traceback
from berny import Berny, geomlib
from berny import optimize
import re
import logging
import io
from contextlib import redirect_stdout

# sys.path.append('./../utils/')

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
utils_dir = os.path.join(script_dir, "../utils/")
sys.path.append(utils_dir)

from isotope_mass import get_atom_symbol
from read_civfl_w_sym import civfl_ana

EV_TO_AU = 1 / 27.21138602
ANG_TO_BOHR = 1.8897259886
PRINT = False


class ParseFile:
    """
    Class to parse COLUMBUS gradients and NAC files
    """

    def __init__(self, fDIR, nroots, QMin):
        self.dir = copy.deepcopy(fDIR)
        self.nroots = nroots
        self.qmin = QMin

    def parse_grad(self):
        _grad = {}
        for _state in range(self.nroots):
            _fname = "%s/GRADIENTS/cartgrd.drt1.state%s.sp" % (self.dir, _state + 1)
            try:
                with open(_fname, 'r') as f:
                    flines = f.readlines()
            except FileNotFoundError:
                print(f"Gradient file {_fname} not found.")
                if 'screening' in self.qmin:
                    print('This is expected from screening, ignore the warning!\n')
                else:
                    sys.exit(1)

            _state_grad = {}
            for _ct, _lines in enumerate(flines):
                _state_grad[_ct] = {'x': float(_lines.split()[0].replace('D', 'E')),
                                    'y': float(_lines.split()[1].replace('D', 'E')),
                                    'z': float(_lines.split()[2].replace('D', 'E'))}

            _grad[_state] = _state_grad
        return _grad

    def parse_NAC(self):
        _nac = {}
        for _state1 in range(self.nroots):
            for _state2 in range(_state1 + 1, self.nroots):
                _fname = "%s/GRADIENTS/cartgrd.nad.drt1.state%s.drt1.state%s.sp" % (self.dir, _state1 + 1, _state2 + 1)
                try:
                    with open(_fname, 'r') as f:
                        flines = f.readlines()
                except FileNotFoundError:
                    print(f"NAC file {_fname} not found.")
                    if 'screening' in self.qmin:
                        print('This is expected from screening, ignore the warning!\n')
                    else:
                        sys.exit(2)

                _state_nac = {}
                for _ct, _lines in enumerate(flines):
                    _state_nac[_ct] = {'x': float(_lines.split()[0].replace('D', 'E')),
                                       'y': float(_lines.split()[1].replace('D', 'E')),
                                       'z': float(_lines.split()[2].replace('D', 'E'))}

                _nac[_state1, _state2] = _state_nac
        return _nac


def grad_mat(_grad, _nac, H0, atom, direction):
    """
    Builds the ZO gradient matrix G^ZO
    :param _grad: gradints for all states (all atoms, all directions)
    :param _nac: nac between all states (all atoms, all directions)
    :param H0: To calculate off diagonal derivative couplings (energy scaled)
    :param atom: which atom to look at
    :param direction: the direction of choice
    :return: Gradient matrix in matrix format: G^ZO in the main paper.
    """
    nstates = len(H0)
    grad = np.zeros([nstates, nstates])
    for _state1 in range(nstates):
        grad[_state1, _state1] = _grad[_state1][atom][direction]
        for _state2 in range(_state1 + 1, nstates):
            grad[_state1, _state2] = _nac[_state1, _state2][atom][direction] * \
                                     -1.0 * (H0[_state2, _state2] - H0[
                _state1, _state1])  # A negative sign is required!!
            grad[_state2, _state1] = grad[_state1, _state2]

    return grad


def get_capmat_H0_opencap(QMin, ref_en=True):
    """
    Parses all CAP files: ZO energies and 1-RDMs
    :param QMin: dict with all QM information
    :param ref_en: If to parse reference (neutral) energies or not. Default is True
    :return: ZO energies, CAP matrix, AO Overlap, pyopencap object
    """
    pathREAD = os.path.join(QMin['newpath'], 'CAP_INPS')
    nstates = int(QMin['nstates'])

    os.chdir(pathREAD)
    sys.stdout.write("\nDensity files are read from path: \n %s \n" % os.getcwd())

    molden_dict = {"basis_file": "molden_mo_mc.sp",
                   "molecule": "molden"}
    s = pyopencap.System(molden_dict)
    cap_dict = {"cap_type": "box",
                "cap_x": "%s" % QMin['cap_x'],
                "cap_y": "%s" % QMin['cap_y'],
                "cap_z": "%s" % QMin['cap_z'],
                "Radial_precision": "16",
                "angular_points": "590",
                "do_numerical": "true"}

    pc = pyopencap.CAP(s, cap_dict, nstates)

    if QMin['calculation'] == 'mcscf':
        parser = colparser_mc('molden_mo_mc.sp')
    else:
        parser = colparser('molden_mo_mc.sp', 'tranls')

    if QMin['calculation'] == 'mcscf':
        H0 = parser.get_H0(filename='%s/mcscfsm' % pathREAD)
        if ref_en:
            ref_energy = parser.get_H0(filename='%s/mcscfsm_neutral' % pathREAD)[0][0]
    else:
        H0 = parser.get_H0('eci', filename='%s/ciudgsm' % pathREAD)
        if ref_en:
            ref_energy = np.min(parser.get_H0('eci', filename='%s/ciudgsm_neutral' % pathREAD))

    if ref_en:
        sys.stdout.write("\nRef energy : %s\n" % ref_energy)

    if QMin['calculation'] == 'mcscf':
        for i in range(0, nstates):
            for j in range(i, nstates):
                if i == j:
                    dm1_ao = parser.sdm_ao(i + 1)
                    pc.add_tdm(dm1_ao, i, j, 'molden')

                else:
                    dm1_ao = parser.tdm_ao(i + 1, j + 1)
                    pc.add_tdm(dm1_ao, i, j, 'molden')
                    pc.add_tdm(dm1_ao.conj().T, j, i, 'molden')
    else:
        for i in range(0, nstates):
            for j in range(i, nstates):
                if i == j:
                    dm1_ao = parser.sdm_ao(i + 1, DRTn=1)
                    pc.add_tdm(dm1_ao, i, j, 'molden')

                else:
                    dm1_ao = parser.tdm_ao(i + 1, j + 1, drtFrom=1, drtTo=1)
                    pc.add_tdm(dm1_ao, i, j, 'molden')
                    pc.add_tdm(dm1_ao.conj().T, j, i, 'molden')

    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])
    _start = datetime.datetime.now()

    ostr = io.StringIO()

    sys.stdout.write("\n\n|%s %s %s|\n\n" % ("=" * 20, "OpenCAP run details:", "=" * 20))
    sys.stdout.write("Calculating CAP matrix via OpenCAP driver. \n")
    sys.stdout.flush()
    with redirect_stdout(ostr):
        pc.compute_projected_cap()
        W = pc.get_projected_cap()
    sys.stdout.write(f"{log_stream.getvalue()}")
    sys.stdout.write(f"{ostr.getvalue()}")
    _end = datetime.datetime.now()
    sys.stdout.write(
        "\n|%s %s %s|\n\n" % ("=" * 20, "End of OpenCAP run (%s Seconds)." % time_string(_start, _end), "=" * 20))
    sys.stdout.flush()

    ao_ovlp = reduce(np.dot, (LA.inv(parser.mo_coeff).T, LA.inv(parser.mo_coeff)))

    CAPH = CAPHamiltonian(H0=H0, W=W)
    CAPH.export("%s/CAPMAT.out" % pathREAD)

    os.chdir(QMin['pwd'])
    return H0, W, ao_ovlp, pc


def _biorthogonalize(Leigvc, Reigvc):
    '''
    Biorthogonalization of Left and right rotation vector
    Adapted from opencap source code.
    '''
    M = Leigvc.T @ Reigvc
    P, L, U = LA.lu(M)
    Linv = LA.inv(L)
    Uinv = LA.inv(U)
    Leigvc = np.dot(Linv, Leigvc.T)
    Reigvc = np.dot(Reigvc, Uinv)
    Leigvc = Leigvc.T
    return Leigvc, Reigvc


def _sort_eigenvectors(eigv, eigvc):
    '''
    Sort eigen vector columns according to the ascending order of
    real part of eigen values.
    :param eigv:
    :param eigvc:
    :return: eigv, eigvc
    '''
    idx = eigv.argsort()
    eigv = eigv[idx]
    eigvc = eigvc[:, idx]
    return eigv, eigvc


def ZO_TO_DIAG_energy(H0, W, eta_opt, corrected=False):
    """
    Diagonalize ZO energy matrix
    :param H0: ZO energy matrix
    :param W: CAP matrix
    :param eta_opt: eta value
    :param corrected: To ask for 1st order corrected energies or not (Default is False)
    :return: complex eigen values and eigen vectors
    """
    H_total = H0 + 1.0j * eta_opt * 0.5 * (W + W.T)
    nstates = len(H0)
    eigv, Reigvc = _sort_eigenvectors(*LA.eig(H_total))

    W = reduce(np.dot, (Reigvc.T, Reigvc))
    W_sqrt = LA.sqrtm(W)
    W_inv_sqrt = LA.inv(W_sqrt)
    Reigvc = reduce(np.dot, (Reigvc, W_inv_sqrt))

    corrected_energies = []
    for i in range(0, nstates):
        total = 0
        for k in range(0, nstates):
            for l in range(0, nstates):
                total += Reigvc[k, i] * W[k][l] * Reigvc[l, i]
        total *= 1.0j
        corrected_energies.append(eigv[i] - eta_opt * total)
    corrected_energies = np.asarray(corrected_energies)

    if corrected:
        return corrected_energies, Reigvc, Reigvc
    else:
        return eigv, Reigvc, Reigvc


def time_string(starttime, endtime):
    runtime = endtime - starttime
    total_seconds = runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds / 1.e6
    return total_seconds


def get_corrected_GMAT(G_MCH, RotMATnew, QMin):
    """
    Get CAP-gradient augmented gradient matrices (G^ZO --> G^DIAG)
    :param G_MCH: ZO Gradient matrix (G^ZO)
    :param RotMATnew: rotation matrix in a new step
    :param QMin: dict with QM info
    :return: Full gradient matrix (G^DIAG)
    """
    pathREAD = os.path.join(QMin['newpath'], 'CAP_INPS')
    Reigvc = RotMATnew['Reigvc']
    Leigvc = RotMATnew['Leigvc']

    os.chdir(pathREAD)
    nstates = QMin['nstates']

    if QMin['calculation'] == 'mcscf':
        parser = colparser_mc('molden_mo_mc.sp')
    else:
        parser = colparser('molden_mo_mc.sp', 'tranls')

    molden_dict = {"basis_file": "molden_mo_mc.sp",
                   "molecule": "molden"}
    s = pyopencap.System(molden_dict)

    cap_dict = {"cap_type": "box",
                "cap_x": "%s" % QMin['cap_x'],
                "cap_y": "%s" % QMin['cap_y'],
                "cap_z": "%s" % QMin['cap_z'],
                "Radial_precision": "16",
                "angular_points": "590", "do_numerical": "true"}

    pCAPG = pyopencap.CAP(s, cap_dict, nstates)

    if QMin['calculation'] == 'mcscf':
        for i in range(0, nstates):
            for j in range(i, nstates):
                if i == j:
                    dm1_ao = parser.sdm_ao(i + 1)
                    pCAPG.add_tdm(dm1_ao, i, j, 'molden')

                else:
                    dm1_ao = parser.tdm_ao(i + 1, j + 1)
                    pCAPG.add_tdm(dm1_ao, i, j, 'molden')
                    pCAPG.add_tdm(dm1_ao.conj().T, j, i, 'molden')
    else:
        for i in range(0, nstates):
            for j in range(i, nstates):
                if i == j:
                    dm1_ao = parser.sdm_ao(i + 1, DRTn=1)
                    pCAPG.add_tdm(dm1_ao, i, j, 'molden')

                else:
                    dm1_ao = parser.tdm_ao(i + 1, j + 1, drtFrom=1, drtTo=1)
                    pCAPG.add_tdm(dm1_ao, i, j, 'molden')
                    pCAPG.add_tdm(dm1_ao.conj().T, j, i, 'molden')

    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])
    ostr = io.StringIO()
    _start = datetime.datetime.now()

    sys.stdout.write("\n\n|%s %s %s|\n\n" % ("=" * 20, "OpenCAP run details:", "=" * 20))
    sys.stdout.write("Calculating gradient contribution from CAP via OpenCAP driver. \n")
    sys.stdout.flush()
    with redirect_stdout(ostr):
        pCAPG.compute_projected_capG()
        _WG = pCAPG.get_projected_capG()
        pCAPG.compute_projected_cap_der()
        _WD = pCAPG.get_projected_cap_der()

    sys.stdout.write(f"{log_stream.getvalue()}")
    sys.stdout.write(f"{ostr.getvalue()}")
    _end = datetime.datetime.now()
    sys.stdout.write(
        "\n|%s %s %s|\n\n" % ("=" * 20, "End of OpenCAP run (%s Seconds)." % time_string(_start, _end), "=" * 20))
    sys.stdout.flush()

    G_DIAG_CORR = []
    _just_grad_correct = []

    for idx, symbol in enumerate(QMin['atom_symbols']):
        if symbol == 'X': continue

        G_DIAG_CORR.append({'x': Leigvc.T @ (
                G_MCH[idx]['x'] + 1.0j * float(QMin['eta_opt']) * (_WG[idx]['x'] + _WD[idx]['x'])) @ Reigvc,
                            'y': Leigvc.T @ (G_MCH[idx]['y'] + 1.0j * float(QMin['eta_opt']) * (
                                    _WG[idx]['y'] + _WD[idx]['y'])) @ Reigvc,
                            'z': Leigvc.T @ (G_MCH[idx]['z'] + 1.0j * float(QMin['eta_opt']) * (
                                    _WG[idx]['z'] + _WD[idx]['z'])) @ Reigvc})
        _just_grad_correct.append(
            {'x': (Leigvc.T @ (1.0j * float(QMin['eta_opt']) * (_WG[idx]['x'] + _WD[idx]['x'])) @ Reigvc).real,
             'y': (Leigvc.T @ (1.0j * float(QMin['eta_opt']) * (_WG[idx]['y'] + _WD[idx]['y'])) @ Reigvc).real,
             'z': (Leigvc.T @ (1.0j * float(QMin['eta_opt']) * (_WG[idx]['z'] + _WD[idx]['z'])) @ Reigvc).real})

    os.chdir(QMin['pwd'])

    return G_DIAG_CORR, _just_grad_correct


def write_positions_to_file(file_path, atom_symbols, positions, header=True):
    lines_to_write = []
    if header:
        lines_to_write = [str(len(atom_symbols))]
        lines_to_write.append("Bohr")

    for symbol, (x, y, z) in zip(atom_symbols, positions):
        line = f"{symbol}  {x:.12f}  {y:.12f}  {z:.12f}"
        lines_to_write.append(line)

    with open(file_path, 'w') as f:
        f.write("\n".join(lines_to_write))


def writefile(filename, content):
    # content can be either a string or a list of strings
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


def getGhostCoord(coords, QMin, geomSUPPL=None, ghidx=1):
    """
    Get ghost atom in Centre-of-mass

    Parameters:
    ----------
    QMin

    geomSUPPL: supplied geometry (Default is None)

    ghidx: ghost atoms count (Default: 1)

    Returns:
    -------
    Ghost atom coordinates
    """

    if geomSUPPL:
        geo = geomSUPPL
    else:
        geo = coords

    atom_symbols = QMin['atom_symbols']

    import isotope_mass
    mass_coord = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0

    for iatom, atom in enumerate(geo):
        if atom_symbols[iatom] == 'X': continue
        total_mass += isotope_mass.MASSES.get(atom_symbols[iatom])
        _coord = np.array([isotope_mass.MASSES.get(atom_symbols[iatom]) * coord for coord in atom])
        mass_coord += _coord

    coord_com = (mass_coord / total_mass)
    return coord_com


def parse_geom_from_daltaoin(path, position, QMin):
    atom_symbols = QMin['atom_symbols']
    daltaoinfile = os.path.join(path, 'daltaoin')
    daltstring = readfile(daltaoinfile)

    atom_symbols_daltaoin = []
    for idx, line in enumerate(daltstring):
        if len(line.split()) and line.split()[-1] == '*':
            splitlines = line.split()
            atom_symbols_daltaoin.append(splitlines[0])

    # Synchronize QMin['atom_symbols'] with atom_symbols_daltaoin if necessary
    if atom_symbols_daltaoin != atom_symbols:
        if QMin['symmetry']:
            extra = [x for i, x in enumerate(atom_symbols_daltaoin) if QMin['atom_symbols'].count(x) < atom_symbols_daltaoin[:i+1].count(x)]
            QMin['atom_symbols'] +=extra
        else:
            QMin['atom_symbols'] = copy.deepcopy(atom_symbols_daltaoin)


    ghostcoord = getGhostCoord(position, QMin)
    position_list = position.tolist()

    for i, symbol in enumerate(QMin['atom_symbols']):
        if symbol == 'X':
            position_list.insert(i, ghostcoord)

    position_ghostadded = np.asarray(position_list)

    return position_ghostadded


def symmetrize_geometry(geom_w_symbols, QMin):
    """
    Symmetrize the input geometry based on the unique geometry. (A very shoddy solution, alas!)

    Uses pyscf.grad.rhf.symmetrize

    Parameters:
        geom_w_symbols (list): Full geometry as a list of [symbol, x, y, z].

    Returns:
        list: Symmetrized geometry.
    """
    from pyscf import gto
    from pyscf.grad.rhf import symmetrize

    for iatom, atom in enumerate(geom_w_symbols):
        if atom[0] == 'X':
            basis = {'X': gto.basis.load('sto3g', 'H')}

    mol = gto.M(unit='bohr', verbose=0, basis=basis)
    mol.atom = geom_w_symbols
    symmetry_col = readfile(os.path.join(QMin['newpath'], 'geom.unik'))[0].strip()
    #mol.symmetry = symmetry_col
    mol.symmetry = True
    mol.build()

    mol.groupname = copy.deepcopy(symmetry_col)
    mol.topgroup = copy.deepcopy(symmetry_col)

    # Symmetrize the geometry
    symmetrized_geom = symmetrize(mol, mol.atom_coords())

    return np.asarray(symmetrized_geom)

def run_unik(path, QMin):
    """
    Run $COLUMBUS/unik.gets.x to generate symmetry unique atoms
    (may be redundant in future)
    :param path: Path to run the program
    :param QMin:
    :return: symmetry unique atoms' geometries
    """

    rotmax = np.eye(3, dtype=int)
    print("**Warning**\n  Not rotating the given geom, rotmax is diagonal with element 1 !\n\n")
    with open(os.path.join(path, 'rotmax'), 'w') as f:
        for row in rotmax:
            f.write(' '.join(map(str, row)) + '\n')

    reuturncode = call_colprog('unik.gets', QMin, workdir=path)
    if reuturncode.get('unik.gets') != 0:
        sys.exit(f"{'unik.gets'} returned error with errorcode: {reuturncode.get('unik.gets')}")
    else:
        unik_flines = readfile(os.path.join(path, 'geom.unik'))
        geom_unik = []
        symbols_unik = []
        for idx, line in enumerate(unik_flines):
            items = line.split()
            if len(items) == 4:
                symbols_unik.append(get_atom_symbol(int(items[0])))
                geom_unik.append([float(i) for i in items[1:]])

        symbols_unik.append(get_atom_symbol(int(items[0])))
        return symbols_unik, np.asarray(geom_unik)


def make_daltaoin(path, new_positions, QMin):
    """
    Make daltaoin file (may be redundant in future)
    $COLUMBUS/hernew.x does the job
    :param path:
    :param new_positions:
    :param QMin:
    :return:
    """

    daltaoinfile = os.path.join(path, 'daltaoin')
    daltstring = readfile(daltaoinfile)

    if QMin['symmetry']:
        symbols_unik, geom_unik = run_unik(path, QMin)
        atom_symbols = symbols_unik
        _positions = geom_unik

    else:
        atom_symbols = QMin['atom_symbols']
        _positions = new_positions

    atom_ct = 0
    for idx, line in enumerate(daltstring):
        if len(line.split()) and line.split()[-1] == '*':
            splitlines = line.split()
            # atom_ct = int(line.split()[1])
            if splitlines[0] == atom_symbols[atom_ct]:
                splitlines[2] = "{:.17f}".format(_positions[atom_ct][0])[:17]
                splitlines[3] = "{:.17f}".format(_positions[atom_ct][1])[:17]
                splitlines[4] = "{:.17f}".format(_positions[atom_ct][2])[:17]
                atom_ct += 1
            else:
                print('Mismatch in supplied daltaoin and geom file ordering')
                sys.exit(123)
            daltstring[idx] = '%s  %s   %s   %s   %s       %s\n' % (splitlines[0], splitlines[1],
                                                                    splitlines[2], splitlines[3],
                                                                    splitlines[4], splitlines[5])

    writefile(daltaoinfile, daltstring)
    return


def make_colfiles(path, new_positions, QMin, fname):
    atom_symbols = QMin['atom_symbols']

    iargfile = os.path.join(path, fname)
    iargosky = readfile(iargfile)
    atom_ct = 0
    for idx, line in enumerate(iargosky):

        if atom_ct < len(atom_symbols):
            if line.startswith(atom_symbols[atom_ct]) and len(iargosky[idx + 1].split()) == 3:
                splitlines = iargosky[idx + 1].split()
                splitlines[0] = "{:.17f}".format(new_positions[atom_ct][0])[:17]
                splitlines[1] = "{:.17f}".format(new_positions[atom_ct][1])[:17]
                splitlines[2] = "{:.17f}".format(new_positions[atom_ct][2])[:17]
                iargosky[idx + 1] = ' %s\t%s\t%s\n' % (splitlines[0], splitlines[1], splitlines[2])
                atom_ct += 1

    writefile(iargfile, iargosky)

    return


def make_geom(path, new_positions, QMin):
    """
    Print new `geom' file.
    :param path:
    :param new_positions: New poistions at current iteration
    :param QMin:
    :return: geom file with new geometries
    """
    geomfile = os.path.join(path, 'geom')
    geomlines = readfile(geomfile)
    atom_symbols = QMin['atom_symbols']
    import isotope_mass

    atom_ct = 0
    for idx, line in enumerate(geomlines):
        splitlines = line.split()
        if splitlines[0] == 'X':
            atom_mass = 0.0
        else:
            atom_mass = isotope_mass.MASSES.get(splitlines[0])

        if splitlines[0] == atom_symbols[atom_ct]:
            splitlines[2] = "{:11.8f}".format(new_positions[atom_ct][0])[:11]
            splitlines[3] = "{:11.8f}".format(new_positions[atom_ct][1])[:11]
            splitlines[4] = "{:11.8f}".format(new_positions[atom_ct][2])[:11]
            splitlines[5] = "{:11.8f}".format(atom_mass)[:11]
            string = ' '
            #            string += '\t'.join(splitlines)+'\n'
            string += splitlines[0]
            string += ' ' * 5 + splitlines[1]
            string += ' ' * 3 + splitlines[2]
            string += ' ' * 3 + splitlines[3]
            string += ' ' * 3 + splitlines[4]
            string += ' ' * 2 + splitlines[5]
            string += '\n'
        else:
            print('Mismatch in supplied coordinates and geom file ordering\n'
                  'If ghost atom is built in daltaoin file use "ghost_atom" keyword in COLUMBUS.template.\n')
            sys.exit(123)

        geomlines[idx] = string
        atom_ct += 1

    writefile(geomfile, geomlines)
    geomfile_restart = os.path.join(QMin['pwd'], 'geom.restart')
    with open(geomfile_restart, 'w') as f:
        for idx, line in enumerate(geomlines):
            splitlines = line.split()
            if splitlines and splitlines[0] != 'X':
                f.write(line)

    return


def make_transmomin(path, QMin):
    """
    Prepares transmomin file for transition moment calculations (1-RDMs)

    We use lower--> upper state print

    (COLUMBUS default: MCSCF: upper-->lower &  CI: lower-->upper)

    :param path:
    :param QMin:
    :return: transmomin file
    """
    transmominfile = os.path.join(path, 'transmomin')
    mcdeninfile = os.path.join(path, 'mcdenin')

    string = ''
    string += 'MCSCF\n'
    for i in range(QMin['nstates']):
        string += '1\t%i\t1\t%i\n' % (i + 1, i + 1)
        for j in range(i + 1, QMin['nstates']):
            string += '1\t%i\t1\t%i\n' % (i + 1, j + 1)
    writefile(transmominfile, string)
    writefile(mcdeninfile, string)

    return


def copy_files(src_dir, dst_dir):
    '''
    Copy files (wihtout using shutil.copytree)
    :param src_dir: Source directory
    :param dst_dir: Destination directory
    :return:
    '''
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        if os.path.isdir(src_item):
            copy_files(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)


def create_inp(position, QMin, iter):
    """
    Set up the input directory for COLUMBUS run
    :param position: New position
    :param QMin:
    :param iter: Current iteration
    :return: A workdir with all input files.
    """
    path = QMin['newpath']

    sys.stdout.write(f'\n\n===> Scratch directory path: {path}\n\n')
    sys.stdout.flush()

    dir_input = QMin.get('inputdir', os.path.join(str(QMin['pwd']), 'copyfiles/'))

    dir_new_NEU = os.path.join(path, 'NEUTRAL')
    dir_new_AN = os.path.join(path, 'ANION')

    os.makedirs(path, exist_ok=True)
    os.makedirs(dir_new_NEU, exist_ok=True)
    os.makedirs(dir_new_AN, exist_ok=True)

    copy_files(dir_input, path)

    if 'ghost_atom' in QMin:
        position_ghostadded = parse_geom_from_daltaoin(path, position, QMin)
    else:
        position_ghostadded = position

    files_to_copy = ["daltaoin", "iargosky", "inpcol", 'geom']

    make_geom(path, position_ghostadded, QMin)
    make_colfiles(path, position_ghostadded, QMin, fname='iargosky')
    make_colfiles(path, position_ghostadded, QMin, fname='inpcol')
    # make_daltaoin(path, position_ghostadded, QMin)
    # ^^ Redundant, cause unik.gets.x and hernew.x automatically does this (daltaoin-->daltaoin.new)!

    # 1. Generate symmetry unique atoms: geom.unik file
    run_unik(path, QMin)
    # 2. Call hernew.x and create daltaoin.new
    reuturncode = call_colprog('hernew', QMin, workdir=path)
    if reuturncode.get('hernew') == 0:
        # 3. copy daltaoin.new to daltaoin
        shutil.copy(os.path.join(path, 'daltaoin.new'), os.path.join(path, 'daltaoin'))
    else:
        make_daltaoin(path, position_ghostadded, QMin)

    if QMin['symmetry']:
        geom_w_symbols = [[symbol] + geo.tolist() for (symbol, geo) in zip(QMin['atom_symbols'], position_ghostadded)]
        symmetrized_geometry = symmetrize_geometry(geom_w_symbols, QMin)
        if not np.allclose(symmetrized_geometry, position_ghostadded):
            sys.stdout.write("\n\n*** Warning: Given geom and symmetrized geoms are not same!\n")
            sys.stdout.write("\tpyscf.grad.rhfsymmetrize module is used, current geometry (before alignment)!\n")
            string = 'Geometry in Bohrs:\n'
            for i in range(len(QMin['atom_symbols'])):
                string += '%s ' % (QMin['atom_symbols'][i])
                for j in range(3):
                    string += '% 16.12f ' % (position_ghostadded[i][j])
                string += '\n'
            sys.stdout.write(string)

            sys.stdout.write("===> Current geometry (after alignment)!\n")
            string = 'Geometry in Bohrs:\n'
            for i in range(len(QMin['atom_symbols'])):
                string += '%s ' % (QMin['atom_symbols'][i])
                for j in range(3):
                    string += '% 16.12f ' % (symmetrized_geometry[i][j])
                string += '\n'
            sys.stdout.write(string)
            sys.stdout.flush()
        QMin['coords_symmetrized'] = True
        QMin['coords'] = copy.deepcopy(symmetrized_geometry)
        make_geom(path, symmetrized_geometry, QMin)

    if QMin['calculation'] == 'mcscf':
        make_transmomin(dir_new_AN, QMin)

    for file in files_to_copy:
        shutil.copy("%s/%s" % (path, file), dir_new_NEU)
        shutil.copy("%s/%s" % (path, file), dir_new_AN)

    if iter > 0 and 'mcscf_guess' in QMin:
        ctrlfile = os.path.join(path, 'control.run')
        keywords(ctrlfile, 'scf')
    return


def modify_ciudgin_tol(path, QMin, iroot=None):
    '''
    Modify the CI tolerant limit of the specified root in ciudgin file
    :param path: str, path where the ciudgin file exists
    :param iroot: int64, root to modify, Default is none
    :param QMin: dict
    :return:
    '''
    ciudgin = os.path.join(path, 'ciudgin')
    with open(ciudgin, 'r') as f:
        flines = f.readlines()

    for idx, line in enumerate(flines):
        if "RTOLCI" in line:
            stripped_line = line.rstrip().split()
            stripped_line = stripped_line[-1].split(',')
            rtolci = []
            try:
                for items in stripped_line:
                    rtolci.append(float(items))
            except:
                pass

    if QMin.get('rtolci'):
        rtolci = QMin['rtolci']

    if iroot:
        while True:
            if rtolci[iroot] < 1.0E-1:
                rtolci[iroot] *= 10.0
                sys.stdout.write(
                    f"\n\n** Warning: Lowering RTOLCI in ciudgin for root {iroot + 1} by 10 fold. **\n")
                sys.stdout.flush()
                break
            else:
                iroot -= 1

    string = ''
    for idx, line in enumerate(flines):
        if "RTOLCI" in line:
            newline = f" RTOLCI = {','.join('%2.0e' % i for i in rtolci)}\n"
            string += newline
        else:
            string += line

    with open(ciudgin, 'w') as f:
        f.write(string)

    QMin['rtolci'] = rtolci


def modify_cigrdin(path, QMin):
    '''
    Changes tolerate limits in cigridin
    :param path:
    :param QMin:
    :return: New cigridin file with modified tolerance
    '''

    def _extract_tols(QMin, fname):
        lines = readfile(fname)
        tol_pattern = re.compile(r"(\w*tol\w*)\s*=\s*([\d.eE+-]+)")

        for line in lines:
            matches = tol_pattern.findall(line)
            for match in matches:
                key, value = match
                QMin[key] = float(value)

    cigrdin = os.path.join(path, 'cigrdin')
    _extract_tols(QMin, cigrdin)

    string = f"& input\n" \
             f"nmiter = 100, print = 0, fresdd = 1,\n" \
             f"fresaa = 1, fresvv = 1,\n" \
             f"mdir = 1,\n" \
             f"cdir = 1,\n" \
             f"rtol = {QMin.get('rtol', 1e-6) * 10.0}, dtol = {QMin.get('dtol', 1e-6) * 10.0},\n" \
             f"wndtol =  {QMin.get('wndtol', 1e-7) * 10.0}, wnatol = {QMin.get('wnatol', 1e-7) * 10.0}, wnvtol =  {QMin.get('wnvtol', 1e-7) * 10.0}\n" \
             f"nadcalc = 3\n" \
             f"samcflag = 0\n" \
             f"& end"
    with open(cigrdin, 'w') as f:
        sys.stdout.write('\n\n** Lowering limits in cigrdin by 10 fold! **\n\n')
        sys.stdout.flush()
        f.write(string)


def call_colprog(prog, QMin, workdir=None):
    """
    Calls available COLUMBUS binaries
    :param prog: program to call (hernew, dalton, tran, cigrd etc.)
    :param QMin:
    :param workdir: Specify the work directory, if not present run in QMin['savedir']
    :return: return code (dict: {prog: errorcode})
    """

    workdir = QMin['savedir'] if workdir is None else workdir

    stdoutfilename = os.path.join(workdir, f'{prog}.out')
    stderrfilename = os.path.join(workdir, f'{prog}.error')
    colpath = QMin.get('columbus', os.environ.get('COLUMBUS'))

    with open(stdoutfilename, 'w') as stdoutfile, open(stderrfilename, 'w') as stderrfile:
        stringrun = [os.path.join(colpath, f'{prog}.x'), '-m', '%s' % (str(QMin['memory'])), '-nproc %s'%QMin['ncpu']]

        try:
            process = sp.run(stringrun, cwd=workdir, stdout=stdoutfile, stderr=stderrfile, shell=False)
            if process.returncode != 0:
                print(f"Error: Program '{prog}' failed with return code {process.returncode}.")
                print(f"Command: {' '.join(stringrun)}")
                print(f"Check the output files:\n  STDOUT: {stdoutfilename}\n  STDERR: {stderrfilename}")
                raise sp.CalledProcessError(process.returncode, stringrun)
        except sp.CalledProcessError as e:
            print(f"Subprocess error occurred! Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            raise
            # Re-raise the exception after logging the error details

    return {prog: process.returncode}


def extract_allkeys(cidict, fname):
    lines = readfile(fname)
    tol_pattern = re.compile(r"(\w*\w*)\s*=\s*([\d.eE+-]+)")

    for line in lines:
        matches = tol_pattern.findall(line)
        for match in matches:
            key, value = match
            cidict[key] = float(value) if 'tol' in key else int(value)


def call_cigrd(QMin):
    """
    Calculates gradients (flow of logic is adapted from $COLUMBUS/runc)
    :param QMin:
    :return: cartgrd file with gradient
    """
    prevdir = os.getcwd()
    path = QMin['savedir']
    os.chdir(path)
    starttime = datetime.datetime.now()
    sys.stdout.write('START:\t%s\t%s \n' % (path, starttime))
    sys.stdout.flush()

    # Extract all variables
    cidict = {}
    cigrdin = os.path.join(path, 'cigrdin')
    extract_allkeys(cidict, cigrdin)
    # Set QMin nadcalc variable manually to zero
    cidict['nadcalc'] = 0

    string = '& input\n'
    for key in cidict:
        string += f" {key}={cidict[key]},\n"
    if not cidict.get('assume_fc'):
        string += f"assume_fc=0\n"
    string += '& end'
    with open(cigrdin, 'w') as f:
        f.write(string)
    # Created the cigrdin file.
    return_codes = []

    # Run the program
    rcode = call_colprog('cigrd', QMin)
    return_codes.append(rcode)

    try:
        shutil.move("effd1fl", "modens")
        shutil.move("effd2fl", "modens2")
    except FileNotFoundError as e:
        raise RuntimeError(f"Rename error: {e}")

    for file in ["cid1fl", "cid2fl"]:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass

    if QMin['calculation'] != 'mcscf':
        transtring = '&input\n denopt=1\n trdens=1\n tr1e=0\n&end'
        with open("tranin", 'w') as f:
            f.write(transtring)
    else:
        pass

    rcode = call_colprog("tran", QMin)
    return_codes.append(rcode)
    try:
        abacsustring = readfile(os.path.join(QMin['newpath'], 'ANION', 'abacusin'))
        abacsustring = "".join(line for line in abacsustring)
    except Exception as e:
        traceback.print_exc()
        print("\n\n** Trying to generate abacusin! It may encounter error in dalton run!**\n\n")
        abacsustring = "**DALTON INPUT \n" \
                       ".PROPERTIES \n" \
                       ".PRINT\n" \
                       "  3\n" \
                       "**ABACUS\n" \
                       ".MOLGRA\n" \
                       ".COLBUS\n" \
                       "*READIN\n" \
                       ".MAXPRI\n" \
                       "  25\n" \
                       "**END OF INPUTS"

    with open('daltcomm', 'w') as f:
        f.write(abacsustring)
    rcode = call_colprog("dalton", QMin)
    return_codes.append(rcode)

    _fname = "%s/GRADIENTS/cartgrd.drt1.state%s.sp" % (QMin['maindir'], QMin['pair'][0])
    shutil.copy('cartgrd', _fname)

    endtime = datetime.datetime.now()
    sys.stdout.write(
        'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %s\n' % (os.getcwd(), endtime, endtime - starttime, return_codes))
    sys.stdout.flush()

    string = ''
    string += ('\n\n' + '--' * 12 + f"Gradient of state {QMin['pair'][0]} " + '--' * 12 + '\n\n')
    lines = readfile('cartgrd')
    for line in lines:
        string += line
    string += '\n'

    os.chdir(prevdir)
    shutil.rmtree(QMin['savedir'], ignore_errors=True)

    return string


def call_nacgrad(QMin):
    """
    Calculates NACs (flow of logic is adapted from $COLUMBUS/runc)
    :param QMin:
    :return: cartgrd file with gradient
    """

    starttime = datetime.datetime.now()
    prevdir = os.getcwd()
    path = QMin['savedir']
    os.chdir(path)

    sys.stdout.write('START:\t%s\t%s \n' % (path, starttime))
    sys.stdout.flush()

    # Extract all variables
    cidict = {}
    cigrdin = os.path.join(path, 'cigrdin')
    extract_allkeys(cidict, cigrdin)
    # Set QMin nadcalc variable to 3
    cidict['nadcalc'] = 3
    _nacpairs = QMin['pairs']

    _string = '& input\n'
    for key in cidict:
        _string += f" {key}={cidict[key]},\n"

    if not cidict.get('assume_fc'):
        _string += f"assume_fc=0\n"
    _string += f' drt1=1\n root1={_nacpairs[0]}\n drt2=1\n root2={_nacpairs[1]}\n'
    _string += '& end'

    with open(cigrdin, 'w') as f:
        f.write(_string)

    return_codes = []
    rcode = call_colprog('cigrd', QMin)
    return_codes.append(rcode)

    try:
        shutil.move("effd1fl", "modens")
        shutil.move("effd2fl", "modens2")
    except FileNotFoundError as e:
        raise RuntimeError(f"Rename error: {e}")

    rcode = call_colprog("tran", QMin)
    return_codes.append(rcode)

    try:
        abacsustring = readfile(os.path.join(QMin['newpath'], 'ANION', 'abacusin'))
        stringreplace = ".COLBUS\n" + ".NONUCG\n"
        abacsustring = (stringreplace if line.startswith('.COLBUS') else line for line in abacsustring)
        abacsustring = "".join(line for line in abacsustring)
    except Exception as e:
        traceback.print_exc()
        print("\n\n** Trying to generate abacusin! It may encounter error in dalton run!**\n\n")
        abacsustring = "**DALTON INPUT \n" \
                       ".PROPERTIES \n" \
                       ".PRINT\n" \
                       "  3\n" \
                       "**ABACUS\n" \
                       ".MOLGRA\n" \
                       ".COLBUS\n" \
                       ".NONUCG\n" \
                       "*READIN\n" \
                       ".MAXPRI\n" \
                       "  25\n" \
                       "**END OF INPUTS"

    with open('daltcomm', 'w') as f:
        f.write(abacsustring)

    # Call NAC main part (CI) calculation
    rcode = call_colprog("dalton", QMin)
    return_codes.append(rcode)

    # Call NAC CSF part calculation
    shutil.copy("cid1trfl", "modens")
    if QMin['calculation'] != 'mcscf':
        transtring = '&input\n denopt=1\n trdens=1\n tr1e=1\n&end'
    else:
        traninflines = readfile("./tranin")
        transtring = ''
        for idx, line in enumerate(traninflines):
            if 'end' in line:
                transtring += ' trdens=1\n tr1e=1\n'
                transtring += line
                break
            else:
                transtring += line

    with open("tranin", 'w') as f:
        f.write(transtring)

    try:
        abacsustring = readfile(os.path.join(QMin['newpath'], 'ANION', 'abacusin.nad'))
        abacsustring = "".join(line for line in abacsustring)
    except Exception as e:
        traceback.print_exc()
        print("\n\n** Trying to generate abacusin.nad! It may encounter error in dalton run!**\n\n")

        abacsustring = "**DALTONINPUT\n" \
                       ".INTEGRALS\n" \
                       ".PRINT\n" \
                       " 2\n" \
                       "**INTEGRALS\n" \
                       ".PRINT\n" \
                       " 2\n" \
                       ".SQHDOL \n" \
                       ".NOSUP \n" \
                       ".NOTWO \n" \
                       "*READIN \n" \
                       ".MAXPRI\n" \
                       " 25\n" \
                       "**END OF INPUTS"
    with open('daltcomm', 'w') as f:
        f.write(abacsustring)

    rcode = call_colprog("tran", QMin)
    return_codes.append(rcode)

    rcode = call_colprog('dalton', QMin)
    return_codes.append(rcode)

    dE = QMin['energy_pairs'][0] - QMin['energy_pairs'][1]
    lines = readfile(f'nadxfl')
    nadx, nady, nadz = [], [], []
    for line in lines:
        line = line.replace('D', 'E').strip()
        line = line.lstrip()

        values = line.split()
        if len(values) >= 3:
            nadx.append(float(values[0]))
            nady.append(float(values[1]))
            nadz.append(float(values[2]))

    dCSF = np.zeros((len(nadx), 3))
    dCSF[:, 0] = nadx
    dCSF[:, 1] = nady
    dCSF[:, 2] = nadz

    lines = readfile('cartgrd')
    nadx, nady, nadz = [], [], []

    for line in lines:
        line = line.replace('D', 'E').strip()
        line = line.lstrip()

        values = line.split()
        if len(values) >= 3:
            nadx.append(float(values[0]))
            nady.append(float(values[1]))
            nadz.append(float(values[2]))

    dCI = np.zeros((len(nadx), 3))
    dCI[:, 0] = nadx
    dCI[:, 1] = nady
    dCI[:, 2] = nadz

    dNAC = np.zeros(np.shape(dCI))
    for idim in range(3):
        dNAC[:, idim] = dCSF[:, idim] + dCI[:, idim] / dE

    with open(f'cartgrd_full', 'w') as f:
        for idim in range(np.shape(dNAC)[0]):
            f.write("\t".join(("% 16.12E" % i).replace('E', 'D') for i in dNAC[idim, :]))
            f.write('\n')

    string = ''
    string += ('\n\n' + '--' * 12 + f'NAC between state {_nacpairs[0]}-{_nacpairs[1]} ' + '--' * 12 + '\n\n')
    lines = readfile('cartgrd_full')
    for line in lines:
        string += line
    string += '\n'

    _fname = "%s/GRADIENTS/cartgrd.nad.drt1.state%s.drt1.state%s.sp" % (QMin['maindir'], _nacpairs[0], _nacpairs[1])
    shutil.copy('cartgrd_full', _fname)

    endtime = datetime.datetime.now()
    sys.stdout.write(
        'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %s\n' % (os.getcwd(), endtime, endtime - starttime, return_codes))
    sys.stdout.flush()
    os.chdir(prevdir)
    shutil.rmtree(QMin['savedir'], ignore_errors=True)
    return string


def process_jobs(QMin1):
    '''
    Process NAC and grad jobs (only these are parallelized)
    :param QMin1:
    :return: log strings with success and failure info
    '''
    if 'grad_branch' in QMin1:
        logstr = call_cigrd(QMin1)
    elif 'nac_branch' in QMin1:
        logstr = call_nacgrad(QMin1)
    return logstr


def run_gradients(joblist, _run_args, QMin):
    """
    Run gradient jobs in parallel
    :param joblist: dictionaries of all parallel jobs
    :param _run_args: dicts of arguments specific for each jobs
    :param QMin:
    :return: log information after parallelized run
    """
    from concurrent.futures import ProcessPoolExecutor
    log_entries = {}

    _max_workers = int(QMin['ncpu'])

    with ProcessPoolExecutor(max_workers=_max_workers) as executor:
        results = executor.map(process_jobs, _run_args)
        for i, jobset in enumerate(joblist):
            if not jobset:
                continue
            for job in jobset:
                log_entries[job] = next(results)
                time.sleep(jobset[job].get('delay', 0))

    with open(os.path.join(QMin['savedir'], 'GRAD.ZO.log'), 'a+') as f:
        f.write(
            '\n\n' + '--' * 10 + f" Printing parallel gradient and coupling calculation: Iteration {QMin['iter']} " + '--' * 10 + '\n\n')

        for i, jobset in enumerate(joblist):
            if not jobset:
                continue
            for job in jobset:
                f.write(log_entries[job])


def create_jobdir(path, joblist):
    """
    Prepare job directories for all parallel jobs (Gradients and NACs)
    :param path:
    :param joblist:
    :return: Prepared individual work directories (set up scratchdir variable in COLUMBUS resources file)
    """
    _files_to_copy = ["mcscfin", "cidrtfl", "mcdrtfl*", "restart",
                      "mcdftfl*", "mcd2fl", "mcd1fl", "mcoftfl*",
                      "hdiagf", "mchess", 'moints', 'mocoef', "daltaoin"]
    import glob
    files_to_copy = []

    path_from_copy = os.path.join(path, 'WORK')
    for filec in _files_to_copy:
        files_to_copy.extend([os.path.basename(fn) for fn in glob.glob(os.path.join(path_from_copy, filec))])

    # Need energies for NAC
    try:
        if QMin['calculation'] == 'mcscf':
            parser = colparser_mc(os.path.join(path, 'MOLDEN', 'molden_mo_mc.sp'))
        else:
            parser = colparser(os.path.join(path, 'MOLDEN', 'molden_mo_mc.sp'), os.path.join(path_from_copy, 'tranls'))
    except:
        if QMin['calculation'] == 'mcscf':
            parser = colparser_mc(os.path.join(QMin['newpath'], 'MOLDEN', 'molden_mo_mc.sp'))
        else:
            parser = colparser(os.path.join(QMin['newpath'], 'MOLDEN', 'molden_mo_mc.sp'),
                               os.path.join(path_from_copy, 'tranls'))

    _energies = parser.get_H0(filename='%s/mcscfsm' % path_from_copy).diagonal() if QMin[
                                                                                        'calculation'] == 'mcscf' else parser.get_H0(
        'eci', filename='%s/ciudgsm' % path_from_copy).diagonal()

    _run_args = []
    subdirs = []

    for ijobset, jobset in enumerate(joblist):
        if not jobset:
            continue
        for job in jobset:
            key = [int(job.split('_')[-1])] if len(job.split('_')) < 3 else [int(i) for i in job.split('_')[1:]]

            os.makedirs(jobset[job]['savedir'], exist_ok=True)
            shutil.copy(os.path.join(path, 'cigrdin'), jobset[job]['savedir'])

            trstring = 'mc' if QMin['calculation'] == 'mcscf' else 'ci'

            shutil.copy(os.path.join(path, f'tran{trstring}denin'), os.path.join(jobset[job]['savedir'], 'tranin'))

            if 'nac_branch' in jobset[job]:
                jobset[job]['pairs'] = key

                jobset[job]['energy_pairs'] = [_energies[key[0] - 1], _energies[key[1] - 1]]
                if QMin['calculation'] != 'mcscf':
                    files_for_nac = ["civout.drt1", "civfl.drt1",
                                     f'cid1fl.trd{key[0]}to{key[1]}', f'cid2fl.trd{key[0]}to{key[1]}',
                                     f"cid1trfl.FROMdrt1.state{key[0]}TOdrt1.state{key[1]}"]
                    files_name_final = ["civout.drt1", "civfl.drt1",
                                        f'cid1fl.tr', f'cid2fl.tr',
                                        f"cid1trfl"]
                else:
                    files_for_nac = [f'mcsd1fl.drt1.st{key[0]:02d}-st{key[1]:02d}',
                                     f'mcsd2fl.drt1.st{key[0]:02d}-st{key[1]:02d}',
                                     f"mcad1fl.drt1.st{key[0]:02d}-st{key[1]:02d}"]
                    files_name_final = [f'cid1fl.tr', f'cid2fl.tr',
                                        f"cid1trfl"]

                copyfiles = files_to_copy + files_for_nac
                copyfilesfinal = files_to_copy + files_name_final

            if 'grad_branch' in jobset[job]:
                jobset[job]['pair'] = key
                if QMin['calculation'] != 'mcscf':
                    files_for_grad = [f'cid1fl.drt1.state{key[0]}', f'cid2fl.drt1.state{key[0]}']
                else:
                    files_for_grad = [f'mcsd1fl.drt1.st{key[0]:02d}', f'mcsd2fl.drt1.st{key[0]:02d}']

                files_name_final = ['cid1fl', 'cid2fl']
                copyfiles = files_to_copy + files_for_grad
                copyfilesfinal = files_to_copy + files_name_final

            for idx, filec in enumerate(copyfiles):
                try:
                    shutil.copy(os.path.join(path_from_copy, filec),
                                os.path.join(jobset[job]['savedir'], copyfilesfinal[idx]))
                except Exception as er:
                    print(f"Failed to move {filec}: {er}")
                    traceback.print_exc()

            _run_args.append(jobset[job])
            subdirs.append(jobset[job]['savedir'])

    return _run_args


def screen_jobs(QMin):
    """
    Screen jobs to reduce number of Gradient and NAC calculations (screening in COLUMBUS resources)
    :param QMin:
    :return: Removed gradient and NAC jobs
    """
    cap_iwfmtpath = os.path.join(QMin['newpath'], 'ANION', 'WORK')
    cap_final_path = os.path.join(QMin['newpath'], 'CAP_INPS')
    os.makedirs(cap_final_path, exist_ok=True)
    generate_dens(QMin, cap_iwfmtpath)

    import glob
    capfiles = glob.glob(os.path.join(cap_iwfmtpath, "*iwfmt")) \
               + [os.path.join(QMin['newpath'], 'MOLDEN', 'molden_mo_mc.sp')]

    if QMin['calculation'] == 'mcscf':
        capfiles += glob.glob(os.path.join(cap_iwfmtpath, "mcscfsm"))
    else:
        capfiles += glob.glob(os.path.join(cap_iwfmtpath, "tranls")) + glob.glob(os.path.join(cap_iwfmtpath, "ciudgsm"))

    for idx, cpfile in enumerate(capfiles):
        try:
            _basename = os.path.basename(cpfile)
            shutil.copy(cpfile, os.path.join(cap_final_path, _basename))
        except Exception as er:
            print(f"Failed to move {_basename}: {er}")
            traceback.print_exc()

    screening_param = QMin.get('screening', 1E-6)
    H0, W, _, _ = get_capmat_H0_opencap(QMin, ref_en=False)
    _, Leigvc, Reigvc = ZO_TO_DIAG_energy(H0, W, float(QMin['eta_opt']), corrected=False)

    removed_g_jobs = []
    removed_nac_jobs = []
    for i in range(QMin['nstates']):
        if abs(Leigvc[i, i] * Reigvc[i, i]) < screening_param:
            removed_g_jobs.append(i + 1)
        for j in range(i + 1, QMin['nstates']):
            if abs(Leigvc[i, j] * Reigvc[i, j]) < screening_param and abs(
                    Leigvc[j, i] * Reigvc[j, i]) < screening_param:
                removed_nac_jobs.append((i + 1, j + 1))

    return removed_g_jobs, removed_nac_jobs


def create_joblist(QMin, path):
    """
    Joblist with all parameters are created (for parallel jobs only)
    :param QMin:
    :param path: In the ANION path only
    :return: Joblist for all parallel jobs.
    """
    QMin['scratchdir'] = QMin.get('scratchdir', os.path.join(QMin['savedir'], 'scratch'))
    sys.stdout.write(f"\n{':' * 5} Scratchdir set to {QMin['scratchdir']} {':' * 5}\n\n")
    sys.stdout.flush()

    nacmap = []
    gradmap = []
    for istate in range(QMin['nstates']):
        gradmap.append(istate + 1)
        for jstate in range(istate, QMin['nstates']):
            nacmap.append((istate + 1, jstate + 1))

    if 'screening' in QMin:
        removed_g_jobs, removed_nac_jobs = screen_jobs(QMin)
        for job in removed_g_jobs:
            print(f'Removed job: grad_{job}\n')

        for job in removed_nac_jobs:
            print(f'Removed job: nac_{job[0]}_{job[1]}\n')

        nacmap = [d for d in nacmap if d not in removed_nac_jobs]
        gradmap = [d for d in gradmap if d not in removed_g_jobs]

    joblist = []

    for grad_state in gradmap:
        QMin1 = copy.deepcopy(QMin)
        QMin1['maindir'] = copy.deepcopy(path)
        QMin1['ncpu'] = 1
        QMin1['memory'] = int(QMin['memory'] / QMin['ncpu'])
        QMin1['delay'] = 0.001

        QMin1[f'grad_{grad_state}'] = []
        QMin1['grad_branch'] = []
        QMin1['savedir'] = os.path.join(QMin['scratchdir'], f'grad_{grad_state}')
        joblist.append({f'grad_{grad_state}': QMin1})

    for nac_state in nacmap:
        istate, jstate = nac_state
        QMin1 = copy.deepcopy(QMin)
        QMin1['maindir'] = copy.deepcopy(path)
        QMin1['ncpu'] = 1
        QMin1['memory'] = int(QMin['memory'] / QMin['ncpu'])
        QMin1['delay'] = 0.001

        if istate == jstate:
            continue
        QMin1[f'nac_{istate}_{jstate}'] = []
        QMin1['nac_branch'] = []
        QMin1['savedir'] = os.path.join(QMin['scratchdir'], f'nac_{istate}_{jstate}')
        joblist.append({f'nac_{istate}_{jstate}': QMin1})

    return joblist


def colrun(path, QMin):
    """
    Run COLUMBUS electronic structure job.

    Uses $COLUMBUS/runc

    :param path:
    :param QMin:
    :return:
    """
    colpath = os.environ.get('COLUMBUS')
    os.environ["ARMCI_DEFAULT_SHMMAX"] = "8192"

    ict = 4
    iroot = QMin['nstates']  # Which root to modify : RTOLCI in ciudgin

    starttime = datetime.datetime.now()
    sys.stdout.write('START:\t%s\t%s \n' % (path, starttime))
    sys.stdout.flush()

    while ict > 0:
        try:
            stdoutfilename = os.path.join(path, 'runls')
            stderrfilename = os.path.join(path, 'runc.error')

            if QMin.get('rtolci'):
                if os.path.basename(path) == 'ANION':
                    modify_ciudgin_tol(path, QMin)

            if os.path.basename(path) == 'ANION':
                sstring = 'mcscf' if QMin['calculation'] == 'mcscf' else 'ciudg'
                ctrlfile = os.path.join(path, 'control.run')
                keywords(ctrlfile, 'slope')
                keywords(ctrlfile, 'nadcoupl')
                keywords(ctrlfile, f'{sstring}mom', keep=True)

                if QMin['calculation'] != 'mcscf':
                    # Update IDEN to print 1- and 2-particle densities
                    ciudgin = os.path.join(path, 'ciudgin')
                    if not os.path.isfile(ciudgin):
                        raise FileNotFoundError(f"File not found: {ciudgin}")
                    flines = readfile(ciudgin)
                    cistring = ''
                    seg_transition = False
                    # Add ciudgin with keywords to enable calculating all 2e DENs.
                    for idx, line in enumerate(flines):
                        if 'IDEN' in line:
                            cistring += ' IDEN = 2\n'
                        elif line.strip() == 'transition':
                            seg_transition = True
                            cistring += line
                        else:
                            cistring += line

                    if not seg_transition:
                        cistring += 'transition\n'
                        for istate in range(QMin['nstates']):
                            for jstate in range(istate, QMin['nstates']):
                                cistring += f'1  {istate + 1}  1  {jstate + 1}\n'

                    with open(ciudgin, 'w') as f:
                        f.write(cistring)
                    # END UPDATE

            elif os.path.basename(path) == 'NEUTRAL':
                if QMin['calculation'] == 'mcscf':
                    QMin_single_state = QMin.copy()
                    QMin_single_state['nstates'] = 1
                    make_transmomin(path, QMin_single_state)

            # Open files using 'with' to ensure they are closed properly
            with open(stdoutfilename, 'w') as stdoutfile, open(stderrfilename, 'w') as stderrfile:
                stringrun = [os.path.join(colpath, 'runc'), '-m', '%s' % (str(QMin['memory'])), '-nproc',
                             '%s' % (str(QMin['ncpu']))]
                process = sp.run(stringrun, cwd=path, stdout=stdoutfile, stderr=stderrfile, shell=False)

                # Check the return code manually
                if process.returncode != 0:
                    raise sp.CalledProcessError(process.returncode, stringrun)

                #####
                conv = True
                if QMin['calculation'] != 'mcscf':
                    runcerrorline = readfile(stderrfilename)
                    uroots = []
                    for idx, line in enumerate(runcerrorline):
                        if 'bummer' in line:
                            continue
                        elif 'CI calculation did not converge' in line:
                            conv = False
                            _uroot = int(line.rstrip().split()[-1])
                            if _uroot not in uroots:
                                uroots.append(_uroot)
                if not conv:
                    if len(uroots) == 0:
                        modify_ciudgin_tol(path, QMin, iroot - 1)
                        iroot -= 1
                    else:
                        for iroot in uroots:
                            sys.stdout.write(f"\n** Warning: CIRoot {iroot} did not converge **\n")
                            sys.stdout.flush()
                            modify_ciudgin_tol(path, QMin, iroot - 1)

                    modify_cigrdin(path, QMin)
                #####

                # If everything went fine (including the runc.error read), break the loop
                if conv:
                    ict = 0
                    endtime = datetime.datetime.now()
                    sys.stdout.write(
                        'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (
                            path, endtime, endtime - starttime, process.returncode))
                    sys.stdout.flush()
                    # Finally start parallel calculation
                    if os.path.basename(path) == 'ANION':
                        sys.stdout.write(
                            '\n\n' + '--' * 20 + 'Starting parallel gradient and coupling calculation' + '--' * 20 + '\n\n')
                        sys.stdout.flush()
                        joblist = create_joblist(QMin, path)
                        _run_args = create_jobdir(path, joblist)
                        run_gradients(joblist, _run_args, QMin)

        except sp.CalledProcessError as erx:
            sys.stdout.write(f"Error while running $COLUMBUS/runc in {path} run. Error code: {erx.returncode}")
            sys.stdout.flush()
            ict -= 1

            if QMin['calculation'] != 'mcscf' and os.path.basename(path) == 'ANION':
                runcerrorline = readfile(stderrfilename)
                uroots = []
                for idx, line in enumerate(runcerrorline):
                    if 'bummer' in line:
                        continue
                    elif 'CI calculation did not converge' in line:
                        _uroot = int(line.rstrip().split()[-1])
                        if _uroot not in uroots:
                            uroots.append(_uroot)

                if len(uroots) == 0:
                    modify_ciudgin_tol(path, QMin, iroot - 1)
                    iroot -= 1
                else:
                    for iroot in uroots:
                        sys.stdout.write(f"\n** Warning: CIRoot {iroot} did not converge **\n")
                        sys.stdout.flush()
                        modify_ciudgin_tol(path, QMin, iroot - 1)

                # Modify cigrdin as well!
                modify_cigrdin(path, QMin)
                # Clean WORK path!
                cleanupWORK(QMin, os.path.join(path, 'WORK'))
                copy_files(os.path.join(QMin['newpath'], 'WORK'), os.path.join(path, 'WORK'))

            if 'mcscf_guess' in QMin:
                try:
                    mocoef_recent = os.path.join(path, 'MOCOEFS', 'mocoef_mc.sp')
                    shutil.copy(mocoef_recent, os.path.join(path, 'mocoef'))
                except Exception as er:
                    print(f"Failed to move {mocoef_recent}: {er}")
                    traceback.print_exc()
            if ict == 0:
                sys.exit('Run failed after 4 attempts!')
            '''
            else:
                sys.stdout.write('\n** Running again...\n')
                sys.stdout.flush()
            '''
    return 0
    # Return 0 if everything went fine


def run_calc(coords, iter, QMin):
    """
    Run 3 sequential calculations

    MCSCF, ANION (CI/MCSCF), NEUTRAL (CI/MCSCF) in that order.
    :param coords:
    :param iter:
    :param QMin:
    :return:
    """
    pathSAVE = QMin['savedir']
    create_inp(coords, QMin, iter)

    pathMCSCF = QMin['newpath']
    pathNEU = os.path.join(QMin['newpath'], 'NEUTRAL')
    pathAN = os.path.join(QMin['newpath'], 'ANION')

    try:
        # Take old mocoef as guess!
        if iter > 0 and 'mcscf_guess' in QMin:
            shutil.copy("%s/mocoef.old" % pathSAVE, "%s/mocoef" % pathMCSCF)

        errorcode = colrun(pathMCSCF, QMin)
        if errorcode != 0:
            print("\nError in initial MCSCF calculation\n")
            sys.exit(111)

        shutil.copy("%s/MOCOEFS/mocoef_mc.sp" % pathMCSCF, "%s/mocoef" % pathNEU)
        shutil.copy("%s/MOCOEFS/mocoef_mc.sp" % pathMCSCF, "%s/mocoef" % pathAN)

        # make an old copy
        shutil.copy("%s/MOCOEFS/mocoef_mc.sp" % pathMCSCF, "%s/mocoef.old" % pathSAVE)
        for pathRUN in [pathAN, pathNEU]:
            try:
                copy_files(os.path.join(pathMCSCF, 'WORK'), os.path.join(pathRUN, 'WORK'))
                ctrfile = os.path.join(pathRUN, 'control.run')
                if QMin['calculation'] != 'mcscf':
                    keywords(ctrfile, 'mcscf')
                    keywords(ctrfile, 'mcscfden')
            except Exception as e:
                print(f"\n Error occurred while generating moving MCSCF WORK files :( \n\t{e}")
                traceback.print_exc()

            errorcode = colrun(pathRUN, QMin)

        if iter == 0:
            if QMin.get('track') == 'wfoverlap':
                shutil.copy("%s/MOCOEFS/mocoef_mc.sp" % pathMCSCF, "%s/mocoef.wfov.old" % pathSAVE)

    except sp.CalledProcessError as e:
        print(f"Error while running $COLUMBUS/runc in run.")
        print(e.output)
        exit(911)

    return errorcode


def print_to_table(title, headers, data, align="center", fname=None):
    """
    Print data to table
    :param title: (str) Title information
    :param headers: list(str) Column headers information
    :param data: list(zip(data)), all data in zipped and list format
    :param align: str, which way to align column "center"(default), left", "right"
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

    if fname:
        with open(fname, 'a+') as fn:
            fn.write(f"\n\n{title.center(10)}\n")
    else:
        print(f"\n{title.center(10)}\n")

    table = tabulate(string, headers=headers, tablefmt='pretty', numalign="center", stralign=align, floatfmt='g')

    if fname:
        with open(fname, 'a+') as fn:
            fn.write(table)
    else:
        print(table)


# Function to calculate the individual gradients for each atom in all directions (x, y, z)
def collectinfo(_iter, QMin):
    """
    Collect all the information of electronic structure run
    :param _iter:
    :param QMin:
    :return: gradients, CAP gradinets, str with complex energies,complex energy
    """
    pathSAVE = QMin['savedir']

    nstates = int(QMin['nstates'])
    eta_opt = float(QMin['eta_opt'])
    atom_symbols = QMin['atom_symbols']

    colOBJ = ParseFile("%s/sp_%s/ANION/" % (pathSAVE, _iter), nstates, QMin)
    _grad = colOBJ.parse_grad()
    _nac = colOBJ.parse_NAC()

    H0, W, ao_ovlp, projcap_object = get_capmat_H0_opencap(QMin)
    H_diag, Leigvc, Reigvc = ZO_TO_DIAG_energy(H0, W, eta_opt, corrected=False)
    G_MCH = {}

    idx = 0
    for atom in range(len(atom_symbols)):
        G_MCH[idx] = {'x': grad_mat(_grad, _nac, H0, atom, 'x'),
                      'y': grad_mat(_grad, _nac, H0, atom, 'y'),
                      'z': grad_mat(_grad, _nac, H0, atom, 'z')}
        idx += 1

    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")
    RotMATnew = {}
    RotMATnew['Reigvc'] = Reigvc
    RotMATnew['Leigvc'] = Leigvc
    RotMATnew['ao_ovlp'] = ao_ovlp

    G_MCH_Xfree = []
    for idx, symbol in enumerate(QMin['atom_symbols']):
        if symbol == 'X': continue
        G_MCH_Xfree.append({'x': G_MCH[idx]['x'],
                            'y': G_MCH[idx]['y'],
                            'z': G_MCH[idx]['z']})

    if 'no_capgrad_correct' in QMin:
        grad_all_state = G_MCH_Xfree.copy()
        _just_grad_correct = {}
        for iatom in range(QMin['natom']):
            _just_grad_correct[iatom] = {'x': np.zeros([QMin['nstates'], QMin['nstates']], dtype=complex),
                                         'y': np.zeros([QMin['nstates'], QMin['nstates']], dtype=complex),
                                         'z': np.zeros([QMin['nstates'], QMin['nstates']], dtype=complex)}
    else:
        grad_all_state, _just_grad_correct = get_corrected_GMAT(G_MCH, RotMATnew, QMin)
        G_MCH_Xfree = G_MCH.copy()

    # generate dets file for WFOVERLAP
    if QMin['symmetry']:
        if QMin['track'] == 'civecs':
            generate_dets_w_sym(QMin)
        else:
            import natorb_utils
            DM_diag = natorb_utils.dm_diag(projcap_object).generate_DM_diag(RotMATnew['Reigvc'], RotMATnew['Leigvc'])
            QMin['1RDM_diag'] = DM_diag.copy()
    else:
        generate_dets(QMin)

    if _iter == 0:
        res_state = int(QMin['act_state'][0]) - 1
        if QMin.get('track') == 'dens':
            generate_ad_den_files(QMin, RotMATnew)
            os.makedirs(os.path.join(QMin['newpath'], 'TRACK'), exist_ok=True)
            shutil.copy(RotMATfile, os.path.join(QMin['newpath'], 'TRACK', 'ADDENS.h5'))
        else:
            detsfile = os.path.join(QMin['newpath'], 'TRACK', 'dets')
            shutil.copy(detsfile, os.path.join(pathSAVE, 'dets.old'))
    else:
        if QMin.get('track') == 'dens':
            res_state, maxo = tracking_with_ad_dens(QMin, RotMATnew)
        elif QMin.get('track') == 'civecs':
            res_state, maxo = tracking_with_civectors(QMin, RotMATnew)
        else:
            res_state, maxo = tracking_with_WFOVERLAP(QMin, RotMATnew, lowdinSVD=False)
        title = "Active states (from SA roots)"
        try:
            headers = ["Root (old)", "Root (new)", "Overlap (Re, Im)"]
            data = list(zip(QMin['act_state'], res_state, [maxo]))
            print_to_table(title, headers, data, fname=os.path.join(QMin['pwd'], 'iteration_details'))
        except Exception as e:
            print(f"\n ====> An error occurred in 'collectinfo' module: \n\t{e}")
            traceback.print_exc()

        QMin['act_state'] = copy.deepcopy(res_state)
        res_state = res_state[0] - 1

    if 'natorb' in QMin:
        sys.stdout.write("\n\n** Printing natural orbitals!\n\n")
        sys.stdout.flush()
        import natorb_utils
        BASFILE = os.path.join(str(QMin['pwd']), 'OPENCAPMD.gbs')
        moldensave = os.path.join(str(QMin['pwd']), 'MOLDEN')
        os.makedirs(moldensave, exist_ok=True)
        geomfile = os.path.join(QMin['newpath'], 'geom')
        dentype = 'mc' if QMin['calculation'] == 'mcscf' else 'ci'
        fmolden = os.path.join(moldensave, f'molden_no_{dentype}_diag.iter%i.state%i.sp' % (_iter, res_state + 1))
        try:
            natorb_utils.natorb(projcap_object, ao_ovlp, RotMATnew, geomfile, QMin, BASFILE,
                                fmolden).print_natorb_molden()
        except Exception as e:
            print(f"\n Error occurred while generating natural orbitals :( \n\t{e}")
            traceback.print_exc()

    with h5py.File(RotMATfile, "a") as f:
        if "Reigvc" in f:
            del f["Reigvc"]
        f.create_dataset("Reigvc", data=Reigvc)
        if "Leigvc" in f:
            del f["Leigvc"]
        f.create_dataset("Leigvc", data=Leigvc)
        if "act_state" in f:
            del f["act_state"]
        f.create_dataset("act_state", data=[res_state])

    gradients = []
    gradients_CAP = []

    for idx in range(QMin['natom']):
        gradients.append({'x': grad_all_state[idx]['x'].diagonal().real[res_state],
                          'y': grad_all_state[idx]['y'].diagonal().real[res_state],
                          'z': grad_all_state[idx]['z'].diagonal().real[res_state]})
        gradients_CAP.append({'x': _just_grad_correct[idx]['x'].diagonal().real[res_state],
                              'y': _just_grad_correct[idx]['y'].diagonal().real[res_state],
                              'z': _just_grad_correct[idx]['z'].diagonal().real[res_state]})

    ostring = "  %i \t  %16.10f \t  %16.10f \t  %i " % (_iter, H_diag.real[res_state],
                                                        2.0 * H_diag.imag[res_state], res_state + 1)

    return gradients, gradients_CAP, ostring, H_diag[res_state]


# SBK added this overlap based algorithm.
def overlap_tracking(QMin, RotMatnew):
    pathSAVE = QMin['savedir']

    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")

    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]

    for lst in (QMin['act_state']):
        _ovlp = np.array(
            [np.dot(RotMATold['Reigvc'][:, lst - 1], RotMatnew['Reigvc'][:, k]) for k in range(QMin['nstates'])])
        cur_max = np.where(abs(_ovlp) == abs(_ovlp).max())[0][0]
        maxo = _ovlp[cur_max]
        act_states.append(cur_max + 1)

    return act_states, maxo


def generate_dets(QMin):
    """
    Generate dets file for the current iteration
    :param QMin:
    :return:
    """
    pathTRACK = os.path.join(QMin['newpath'], 'TRACK')
    os.makedirs(pathTRACK, exist_ok=True)

    if QMin['calculation'] == 'mcscf':
        work_files = ['mcdrtfl', 'restart', 'mcscfls']
        script = 'read_mcdrtfl.py'
    else:
        work_files = ['cidrtfl', 'civfl', 'civout']
        script = 'read_civfl.py'

    for file in work_files:
        shutil.copy(os.path.join(QMin['newpath'], 'ANION', 'WORK', file), pathTRACK)

    if QMin['calculation'] == 'mcscf':
        command = [os.path.join(utils_dir, script), str(QMin['nstates'])]
    else:
        command = [f"{QMin['wfovdir']}/scripts/{script}", str(QMin['nstates'])]

    if QMin['calculation'] != 'mcscf':
        command.extend([str(QMin.get('maxsqnorm', '')), "-ms", "0.5"])
    if 'debug' in QMin:
        command.append('-debug')

    try:
        result = sp.run(command, cwd=pathTRACK, check=True, shell=False, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        if 'debug' in QMin:
            print("Output:\n", result.stdout)
            print("Error:\n", result.stderr)
    except sp.CalledProcessError as e:
        print(f"\n[{' '.join(command)}] failed with return code {e.returncode}\n")
        print("Error Output:\n", e.stderr)
        print("Standard Output:\n", e.stdout)


def align_geometry_Kabsch(oldgeom, newgeom, QMin):
    """
    Aligning two sets of points using translation and rotation minimization based on the Kabsch algorithm.
    New geometry is aligned with the old one.

    Parameters:
    ----------
        oldgeom: numpy.ndarray
         Old geometry
        newgeom: numpy.ndarray
         New geometry

    Returns:
    ----------
        aligned_geom: numpy.ndarray
         New aligned geometry
    """

    def centroid(coords):
        """Calculate the centroid of a set of points."""
        return np.mean(coords, axis=0)

    def kabsch_rotation(P, Q):
        """Find the optimal rotation matrix using the Kabsch algorithm."""
        # Compute covariance matrix
        C = np.dot(P.T, Q)
        # Singular Value Decomposition (SVD)
        V, S, W = LA.svd(C)
        # Compute rotation matrix
        d = np.sign(LA.det(np.dot(W.T, V.T)))
        D = np.diag([1, 1, d])
        U = np.dot(W.T, np.dot(D, V.T))
        return U

    def align_geometries(P, Q):
        """Align New (Q) to Old (P) using translation and rotation."""
        # Center both geometries to their centroids
        centroid_P = centroid(P)
        centroid_Q = centroid(Q)
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # Find the optimal rotation matrix
        U = kabsch_rotation(P_centered, Q_centered)

        # Rotate the centered Q geometry
        Q_rotated = np.dot(Q_centered, U)

        # Translate Q_rotated to the centroid of P
        Q_aligned = Q_rotated + centroid_P

        return Q_aligned

    atom_symbols = QMin['atom_symbols']
    # Align geom2 to geom1
    aligned_newgeom = align_geometries(oldgeom, newgeom)

    # Print the aligned geometry
    print("===> Kabsch algorithm is invoked, current geometry (before alignment)!\n")
    string = 'Geometry in Bohrs:\n'
    for i in range(len(atom_symbols)):
        string += '%s ' % (atom_symbols[i])
        for j in range(3):
            string += '% 7.4f ' % (newgeom[i][j])
        string += '\n'
    print(string)

    print("===> Current geometry (after alignment)!\n")
    string = 'Geometry in Bohrs:\n'
    for i in range(len(atom_symbols)):
        string += '%s ' % (atom_symbols[i])
        for j in range(3):
            string += '% 7.4f ' % (aligned_newgeom[i][j])
        string += '\n'
    print(string)

    return aligned_newgeom


def setupWFOVERLAPDIR(QMin):
    '''
    Sets up working directory, prepare files for wave function overlap wfoverlap.x

    dict: QMin dictionary holding all information.
    Returns:
        A working directory with all necessary files for wfoverlap.x run
    '''

    pathTRACK = os.path.join(QMin['newpath'], 'TRACK')
    pathSAVE = QMin['savedir']
    os.makedirs(pathTRACK, exist_ok=True)

    # Copy MOCOEFs
    try:
        mocoeffile = os.path.join(QMin['newpath'], 'ANION', 'MOCOEFS', 'mocoef_mc.sp')
        shutil.copy(mocoeffile, pathTRACK)
    except:
        mocoeffile = os.path.join(QMin['newpath'], 'MOCOEFS', 'mocoef_mc.sp')
        shutil.copy(mocoeffile, pathTRACK)

    mocoefoldfile = os.path.join(pathSAVE, 'mocoef.wfov.old')
    shutil.copy(mocoefoldfile, pathTRACK)
    # Copy old dets file
    detsoldfile = os.path.join(pathSAVE, 'dets.old')
    shutil.copy(detsoldfile, pathTRACK)

    # Initial string
    string = '''
        ao_read=2
        a_mo=mocoef.wfov.old
        b_mo=mocoef_mc.sp
        a_det=dets.old
        b_det=dets
        a_mo_read=0
        b_mo_read=0
        mix_aoovl=./aoints
        '''

    # First run double_mol with dalton.x (align geometry by kabsch)
    run_double_mol(QMin)
    inputfile = os.path.join(pathTRACK, 'wfov.in')

    rerunWFOVLP = False
    max_retries = 2
    attempts = 0

    starttime = datetime.datetime.now()
    sys.stdout.write('==> Performing WF-OVERLAP calculation:\n')
    while True:
        sys.stdout.write('START:\t%s\t%s \n' % (pathTRACK, starttime))
        sys.stdout.flush()

        if rerunWFOVLP:
            string = string.replace('ao_read=2', 'ao_read=-1').replace('mix_aoovl=./aoints', 'same_aos')

        writefile(inputfile, string)

        debug = 'debug' in QMin
        runerror = runWFOVERLAPS(pathTRACK, QMin['wfoverlap'], QMin['memory'], QMin['ncpu'], DEBUG=debug)

        endtime = datetime.datetime.now()
        sys.stdout.write(
            'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (pathTRACK, endtime, endtime - starttime, runerror))
        sys.stdout.flush()

        # Check Wfoverlap matrix if overlap is alright
        with open(os.path.join(QMin['newpath'], 'TRACK', "wfov.out")) as f:
            out = f.readlines()

        allSTATE_ovlp = wfoverlap_mat(out, "Overlap", QMin)
        # If overlap matrix is all zeros
        if np.allclose(allSTATE_ovlp, np.zeros([QMin['nstates'], QMin['nstates']]), 1E-5):
            print("\n\n===> Warning: WFOVERLAP run yielded Overlap matrix with all zero elements\n"
                  "Possible bug with mix_aoovl, performing same_aos calculation\n\n")
            rerunWFOVLP = True
            attempts += 1

            if attempts >= max_retries:
                print("Reached max retry attempts. Exiting loop.")
                break
        else:
            rerunWFOVLP = False
            break

    # Store dets file as old file for next run
    detsnewfile = os.path.join(pathTRACK, 'dets')
    shutil.copy(detsnewfile, os.path.join(pathSAVE, 'dets.old'))
    shutil.copy(mocoeffile, os.path.join(pathSAVE, 'mocoef.wfov.old'))

    return


def run_double_mol(QMin):
    """
    Runs doublemol columbus job ($COLUMBUS/daltaoin.x).

    :param QMin: dictionary,  containing all QM run info
    :return: A directory set-up for DOUBLEMOL calculation
    """
    pathTRACK = os.path.join(QMin['newpath'], 'TRACK')
    prevdir = os.getcwd()
    os.chdir(pathTRACK)

    if 'ghost_atom' in QMin:
        geoold = parse_geom_from_daltaoin(os.path.join(QMin['oldpath'], 'WORK'), QMin['oldgeom'], QMin)
        geonew = parse_geom_from_daltaoin(os.path.join(QMin['newpath'], 'WORK'), QMin['newgeom'], QMin)
    else:
        geoold = QMin['oldgeom'].copy()
        geonew = QMin['newgeom'].copy()

    # Get aligned geometry (in each iteration)
    aligned_geom = align_geometry_Kabsch(geoold, geonew, QMin)
    # Create daltaoin backup
    shutil.copy(os.path.join(QMin['newpath'], 'WORK', 'daltaoin'),
                os.path.join(QMin['newpath'], 'WORK', 'daltaoin.orig'))
    # Create daltaoin with new geometry (aligned)
    make_daltaoin(os.path.join(QMin['newpath'], 'WORK'), aligned_geom, QMin)

    command = ['%s/scripts/dalton_double-mol.py' % (str(QMin['wfovdir'])), QMin['newpath'], QMin['oldpath'], 'run']
    try:
        result = sp.run(command, cwd=pathTRACK, check=True, shell=False,
                        stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        if 'debug' in QMin:
            print("Output:\n", result.stdout)
            print("Error:\n", result.stderr)
    except sp.CalledProcessError as e:
        print(f"\n[{' '.join(command)}] failed with return code {e.returncode}\n")
        print("Error Output:\n", e.stderr)
        print("Standard Output:\n", e.stdout)
    os.chdir(prevdir)
    return


def runWFOVERLAPS(WORKDIR, wfoverlaps, memory=1000, ncpu=1, DEBUG=False):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = wfoverlaps + ' -m %i' % (memory) + ' -f wfov.in'
    stdoutfile = open(os.path.join(WORKDIR, 'wfov.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'wfov.err'), 'w')
    os.environ['OMP_NUM_THREADS'] = str(ncpu)

    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (WORKDIR, starttime, string))
        sys.stdout.flush()
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
    except OSError:
        print('Call have had some serious problems:', OSError)
        sys.exit(85)

    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write(
            'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (WORKDIR, endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)
    return runerror


def lowdin(S):
    '''Uses Lowdin orthogonalization'''
    _U, _D, _VT = np.linalg.svd(S, full_matrices=True)
    S_ortho = _U @ _VT
    return S_ortho


def tracking_with_WFOVERLAP(QMin, RotMATnew, lowdinSVD=False):
    """
    Tracks a root using WF-OVERLAP software (https://github.com/felixplasser/wfoverlap)
    to evaluate the wavefunction overlaps.

    Parameters:
    ----------
    QMin : dict
        Dictionary containing QMin parameters, including electronic state information.
    RotMATnew : dict
        Dictionary containing rotation matrices for the current step.
    lowdinSVD : bool, optional (default=False)
        If True, applies Löwdin symmetric orthogonalization using singular value decomposition (SVD)
        to the overlap matrix before determining state tracking.

    Returns:
    ----------
    act_states : int
        Index (starting at 1) of the electronic state with the highest overlap.
    maxo : float
        Maximum overlap value corresponding to the tracked state.

    :rtype: tuple (int, float)
    """
    # Set up workdir and generate the wfov.out
    setupWFOVERLAPDIR(QMin)
    pathSAVE = QMin['savedir']

    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")

    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]

    with open(os.path.join(QMin['newpath'], 'TRACK', "wfov.out")) as f:
        out = f.readlines()

    allSTATE_ovlp = wfoverlap_mat(out, "Orthonormalized overlap", QMin)
    if lowdinSVD:
        allSTATE_ovlp = lowdin(allSTATE_ovlp)

    _LeigVc_old = RotMATold['Leigvc']
    _ReigVc_new = RotMATnew['Reigvc']

    allSTATE_ovlp_DIAG = reduce(np.dot, (_LeigVc_old.T, allSTATE_ovlp, _ReigVc_new))
    if lowdinSVD:
        allSTATE_ovlp_DIAG = lowdin(allSTATE_ovlp_DIAG)

    for lst in (QMin['act_state']):
        cur_max = np.where(abs(allSTATE_ovlp_DIAG[lst - 1, :]) == abs(allSTATE_ovlp_DIAG[lst - 1, :]).max())[0][0]
        act_states.append(cur_max + 1)
        maxo = allSTATE_ovlp_DIAG[:, lst - 1][cur_max]

    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
    if 'debug' in QMin:
        print('File created:\t==>\t%s' % (RotMATfile))

    return act_states, maxo


def wfoverlap_mat(out, matrix_type, QMin):
    '''Reads WF-OVERLAP from wfov.out'''
    nroots = int(QMin['nstates'])
    _line_idx = None
    for i, line in enumerate(out):
        if f"{matrix_type} matrix <PsiA_i|PsiB_j>" in line:
            _line_idx = i + 2
            break

    if _line_idx is not None:
        matrix_lines = out[_line_idx:_line_idx + nroots]
        matrix = np.array([[float(val) for val in line.split()[2:nroots + 2]] for line in matrix_lines])
        return matrix
    else:
        print("\n==> Wf-Overlap matrix not found!\n\n")
        return None


def generate_dets_w_sym(QMin):
    '''Genrates dets file (for symmetry on systems)'''
    pathTRACK = os.path.join(QMin['newpath'], 'TRACK')
    os.makedirs(pathTRACK, exist_ok=True)

    starttime = datetime.datetime.now()
    sys.stdout.write('START:\t%s\t%s\t Message: %s\n' % (pathTRACK, starttime, "Generating dets with cipc.x"))
    sys.stdout.flush()

    prevdir = os.getcwd()
    os.chdir(pathTRACK)

    work_files = ['cidrtfl', 'civfl', 'civout', 'ciudgls']

    for file in work_files:
        try:
            shutil.copy(os.path.join(QMin['newpath'], 'ANION', 'WORK', file), pathTRACK)
        except:
            pass
    debug = True if 'debug' in QMin else False
    maxsqnorm = float(QMin.get('maxsqnorm', 2.0))
    ca = civfl_ana(maxsqnorm, debug)
    ms = 0.5
    wname = 'dets'

    for istate in range(1, QMin['nstates'] + 1):
        ca.call_cipc(istate, ms=ms, mem=int(QMin['memory']))
    ca.write_det_file(QMin['nstates'], wname=wname)

    os.chdir(prevdir)
    endtime = datetime.datetime.now()
    sys.stdout.write(
        'FINISH:\t%s\t%s\tRuntime: %s\tMessage: %s\n' % (pathTRACK, endtime, endtime - starttime, 'Generation Done!'))
    sys.stdout.flush()


def read_civecs_SD(finp):
    '''Read Ci vectors coeffs and slater determinants'''
    with open(finp, 'r') as f:
        flines = f.readlines()
    slaters = {}
    for idx, line in enumerate(flines[1:]):
        key = line.split()[0]
        slaters[key] = [float(i) for i in line.split()[1:]]
    return slaters


def generate_civecs_diag(civcectors, eigvector):
    '''Rotates ZO civectors to DIAG'''
    civectors_diag = []
    for idx, civecs in enumerate(civcectors):
        d_ci = []
        for k in range(len(eigvector)):
            d_ci_store = 0.0
            for i in range(len(eigvector)):
                d_ci_store += eigvector[i, k] * civecs[i]
            d_ci.append(d_ci_store)
        civectors_diag.append(d_ci)
    return np.asarray(civectors_diag)


def _sort_slaterdet_key(QMin, slaterd, slaterdnew):
    for key in slaterd:
        if not slaterdnew.get(key):
            slaterdnew[key] = [1E-12] * QMin['nstates']

    for key in slaterdnew:
        if not slaterd.get(key):
            slaterd[key] = [1E-12] * QMin['nstates']

    civc_renewed = []
    civcnew_renewed = []
    for key in slaterd:
        civc_renewed.append(slaterd[key])
        civcnew_renewed.append(slaterdnew[key])
    civc_renewed = np.asarray(civc_renewed)
    civcnew_renewed = np.asarray(civcnew_renewed)

    return civc_renewed, civcnew_renewed


def tracking_with_civectors(QMin, RotMATnew):
    '''Tracking with civectors only (assume that MO basis are same in different iterations)'''
    sys.stdout.write("\n\n==> Symmetry is on, hence no WF-OVERLAP is available\n"
                     "Tracking via ci-vectors overlap!\n\n")
    sys.stdout.flush()

    pathSAVE = QMin['savedir']
    pathTRACK = os.path.join(QMin['newpath'], 'TRACK')
    detsoldfile = os.path.join(pathSAVE, 'dets.old')
    detsnewfile = os.path.join(pathTRACK, 'dets')

    _slaterd = read_civecs_SD(detsoldfile)
    _slaterdnew = read_civecs_SD(detsnewfile)

    civc, civcnew = _sort_slaterdet_key(QMin, _slaterd, _slaterdnew)

    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")

    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]

    _ReigVc_old = RotMATold['Reigvc']
    _ReigVc_new = RotMATnew['Reigvc']

    civectors_diag = generate_civecs_diag(civc, _ReigVc_old)
    civectors_diag_new = generate_civecs_diag(civcnew, _ReigVc_new)

    for lst in (QMin['act_state']):
        allstate_civectors_ovlp = np.array(
            [np.dot(civectors_diag[:, lst - 1], civectors_diag_new[:, k]) for k in range(len(_ReigVc_new))])

        cur_max = np.where(abs(allstate_civectors_ovlp) == abs(allstate_civectors_ovlp).max())[0][0]
        act_states.append(cur_max + 1)
        maxo = allstate_civectors_ovlp[cur_max]

    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])

    shutil.copy(detsnewfile, os.path.join(pathSAVE, 'dets.old'))

    return act_states, maxo


def generate_dens(QMin, work_dir):
    """
    Generate density files in ASCII form. Reads COLUMBUS binary file and convers them using $COLUMBUS/iwfmt.x
    :param QMin:
    :param work_dir: The directory where within the WORK directory lies (ANION)
    :return: 1-RDMs in ASCII format
    """
    import glob
    colpath = QMin.get('columbus', os.environ.get('COLUMBUS'))
    if not colpath:
        raise ValueError("COLUMBUS path not found in QMin or environment variables!")

    def run_iwfmt(work_dir, colpath):
        iwfmt_files = glob.glob(os.path.join(work_dir, "*.iwfmt"))
        for file in iwfmt_files:
            os.remove(file)

        pattern_files = glob.glob(os.path.join(work_dir, "*d1fl.*")) + glob.glob(
            os.path.join(work_dir, "*d1trfl.*")) + glob.glob(os.path.join(work_dir, "aoints"))
        for file_path in pattern_files:
            if os.path.isfile(file_path):
                file_name = os.path.basename(file_path)

                iwfmtstr = f"{file_name}\n1\n"

                iwfmt_file = f"{file_name}.iwfmt"
                iwfmt_path = os.path.join(work_dir, iwfmt_file)
                err_ignore_path = os.path.join(work_dir, "err.ignore")

                with open(iwfmt_path, "w") as out_file, open(err_ignore_path, "w") as err_file:
                    command = [os.path.join(colpath, "iwfmt.x")]
                    sifs = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, cwd=work_dir)
                    sifsout, sifserr = sifs.communicate(iwfmtstr.encode('utf-8'))
                    out_file.write(sifsout.decode('utf-8') if sifsout else "")
                    err_file.write(sifserr.decode('utf-8') if sifserr else "")

    run_iwfmt(work_dir, colpath)


def spectral_norm(A):
    '''Calculates spectral norm of matrix: A'''
    lambdas, _ = LA.eig(A @ A.T.conj())
    lambdamax = lambdas[np.where(abs(lambdas) == np.max(abs(lambdas)))[0][0]]
    if lambdamax < 0.0:
        print(f"Spectral norm {lambdamax} less than zero encountered!")
    return np.sqrt(lambdamax)

def generate_ad_den_files(QMin, RotMATnew, writeTOfile=True):
    '''
    Genrates attachement and detachment densities
    :param QMin:
    :param RotMATnew: New rotation matrices
    :param writeTOfile: (bool) save densities to PCAP.RotMatOld.h5 file or not (Default is True)
    :return:
    '''
    pathSAVE = QMin['savedir']

    generate_dens(QMin, os.path.join(QMin.get('newpath'), 'NEUTRAL', 'WORK'))
    parser_tmp = colparser_mc(f"{os.path.join(QMin.get('newpath'), 'CAP_INPS', 'molden_mo_mc.sp')}")
    dAOneu_new = np.zeros(np.shape(parser_tmp.mo_coeff))
    sstring = 'mcs' if QMin['calculation'] == 'mcscf' else 'ci'
    ststring = '01' if QMin['calculation'] == 'mcscf' else '1'
    ststr = 'st' if QMin['calculation'] == 'mcscf' else 'state'

    parser_tmp.read_iwfmt(dAOneu_new,
                          filen=f"{os.path.join(QMin.get('newpath'), 'NEUTRAL', 'WORK')}/{sstring}d1fl.drt1.{ststr}{ststring}.iwfmt")
    dAOneu_new = parser_tmp.mo_coeff @ dAOneu_new @ parser_tmp.mo_coeff.T

    dm_diag = QMin['1RDM_diag']
    istate = QMin['act_state'][0] - 1
    deltaD_new = reduce(np.dot, (dm_diag[istate, istate] - dAOneu_new, RotMATnew['ao_ovlp']))
    k1, w1 = _sort_eigenvectors(*LA.eig(deltaD_new))

    k_detach1 = k1.copy()
    k_attach1 = k1.copy()

    k_detach1[k1 > 0] = 0
    k_attach1[k1 < 0] = 0

    detach_mo_new = w1 @ np.diag(k_detach1) @ w1.T
    attach_mo_new = w1 @ np.diag(k_attach1) @ w1.T

    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")
    if writeTOfile:
        with h5py.File(RotMATfile, "a") as f:
            if "attach_mo" in f:
                del f["attach_mo"]
            f.create_dataset("attach_mo", data=attach_mo_new)
            if "detach_mo" in f:
                del f["detach_mo"]
            f.create_dataset("detach_mo", data=detach_mo_new)


def tracking_with_ad_dens(QMin, RotMATnew):
    """
    Tracking with attachement and detachment densities.

    Parameters:
    ----------
    QMin : dict
        Dictionary containing QMin parameters, including electronic state information.
    RotMATnew : dict
        Dictionary containing rotation matrices for the current step.

    Returns:
    ----------
    act_states : int
        Index (starting at 1) of the electronic state with the highest overlap.
    maxo : float
        Maximum overlap value corresponding to the tracked state.

    :rtype: tuple (int, float)

    Reference:
    ----------
    Head-Gordon, Martin, et al. The Journal of Physical Chemistry 99.39 (1995): 14261-14270.

    """

    pathSAVE = QMin['savedir']
    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")

    RotMATold = {}
    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]
        attach_mo = f["attach_mo"][:]
        detach_mo = f["detach_mo"][:]

    generate_dens(QMin, os.path.join(QMin.get('newpath'), 'NEUTRAL', 'WORK'))
    parser_tmp = colparser_mc(f"{os.path.join(QMin.get('newpath'), 'CAP_INPS', 'molden_mo_mc.sp')}")
    dAOneu_new = np.zeros(np.shape(parser_tmp.mo_coeff))

    sstring = 'mcs' if QMin['calculation'] == 'mcscf' else 'ci'
    ststring = '01' if QMin['calculation'] == 'mcscf' else '1'
    ststr = 'st' if QMin['calculation'] == 'mcscf' else 'state'

    parser_tmp.read_iwfmt(dAOneu_new,
                          filen=f"{os.path.join(QMin.get('newpath'), 'NEUTRAL', 'WORK')}/{sstring}d1fl.drt1.{ststr}{ststring}.iwfmt")
    dAOneu_new = parser_tmp.mo_coeff @ dAOneu_new @ parser_tmp.mo_coeff.T

    dm_diag = QMin['1RDM_diag']

    maxo = -np.inf
    cur_max = -1
    for istate in range(QMin['nstates']):
        deltaD_new = reduce(np.dot, (dm_diag[istate, istate] - dAOneu_new, RotMATnew['ao_ovlp']))
        k1, w1 = _sort_eigenvectors(*LA.eig(deltaD_new))

        k_detach1 = k1.copy()
        k_attach1 = k1.copy()

        k_detach1[k1 > 0] = 0
        k_attach1[k1 < 0] = 0

        detach_mo_new = w1 @ np.diag(k_detach1) @ w1.T
        attach_mo_new = w1 @ np.diag(k_attach1) @ w1.T

        ddm_diff = spectral_norm(attach_mo_new - attach_mo - detach_mo_new + detach_mo)
        if 1.0 - 0.5 * ddm_diff > maxo:
            cur_max = istate + 1
            maxo = copy.deepcopy(1.0 - 0.5 * ddm_diff)
            attach_mo_act_state = copy.deepcopy(attach_mo_new)
            detach_mo_act_state = copy.deepcopy(detach_mo_new)

    with h5py.File(RotMATfile, "a") as f:
        if "attach_mo" in f:
            del f["attach_mo"]
        f.create_dataset("attach_mo", data=attach_mo_act_state)
        if "detach_mo" in f:
            del f["detach_mo"]
        f.create_dataset("detach_mo", data=detach_mo_act_state)

    os.makedirs(os.path.join(QMin['newpath'], 'TRACK'), exist_ok=True)
    shutil.copy(RotMATfile, os.path.join(QMin['newpath'], 'TRACK', 'ADDENS.h5'))

    return [cur_max], maxo


def keywords(QMinFileName, strlook, stringextra=None, keep=False):
    """
    Finds a specific string in a file and removes it if found.
    If the string is not found and `add_if_missing` is True, it adds the string at the end.

    Parameters:
    QMinFileName (str): Path to the file.
    strlook (str): String to find, remove, or add.
    keep (bool): Whether to add the string if it is found/not found. Default is False.
    stringextra (str): If something extra is to be added
    """
    with open(QMinFileName, 'r+') as file:
        lines = file.readlines()
        file.seek(0)

        # Check if strlook exists
        if any(line.startswith(strlook) for line in lines):
            if not keep:
                # Remove strlook
                file.writelines(line for line in lines if not line.startswith(strlook))
            else:
                if stringextra:
                    stringreplace = strlook + stringextra
                    file.writelines(stringreplace if line.startswith(strlook) else line for line in lines)
                else:
                    file.writelines(lines)
        else:
            file.writelines(lines)  # Write back the original lines
            if keep:
                print(f'{strlook} keyword was missing')
                if not strlook.endswith('\n'):
                    if stringextra:
                        strlook += stringextra
                    strlook += '\n'
                file.write(strlook)
        file.truncate()


def print_matrix_table_string(iter, gradients, header, QMin, dictionary=True):
    table_string = '%s = %s \n' % (header, iter)
    table_string += '=' * 61 + '\n'
    headers = ['Atom', 'x', 'y', 'z']
    table_string += '{:6} | {:15} | {:15} | {:15}\n'.format(*headers)
    table_string += '-' * 61 + '\n'

    atom_symbols = QMin['atom_symbols']

    if dictionary:
        for atom_index, grad_values in gradients.items():
            atom = '%s' % atom_symbols[atom_index]
            x = "{:.10f}".format(grad_values['x'])
            y = "{:.10f}".format(grad_values['y'])
            z = "{:.10f}".format(grad_values['z'])
            table_string += '{:6} | {:15} | {:15} | {:15}\n'.format(atom, x, y, z)
    else:
        for atom_index, grad_values in enumerate(gradients):
            atom = '%s' % atom_symbols[atom_index]
            x = "{:.10f}".format(grad_values[0])
            y = "{:.10f}".format(grad_values[1])
            z = "{:.10f}".format(grad_values[2])
            table_string += '{:6} | {:15} | {:15} | {:15}\n'.format(atom, x, y, z)
    table_string += '=' * 61 + '\n\n'
    return table_string


def print_string(info, elements, matrix):
    float_format = "{:12.8f}"
    ostring = str(info) + "\n"
    headers = ['Atom', 'X', 'Y', 'Z']
    table_data = []
    for atom_index, atom_matrix in enumerate(matrix, start=1):
        formatted_geom = [float_format.format(_matrix) for _matrix in atom_matrix]
        table_data.append([elements[atom_index - 1]] + formatted_geom)
    ostring += tabulate(table_data, headers=headers, tablefmt="pretty")
    return ostring


def print_to_file(ostring, pathRUN):
    with open("%s/iteration_details" % pathRUN, "a+") as f:
        f.write(ostring + "\n")
    return


def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out


def generate_restartfiles(QMin, _iter):
    """
    Genarate restart files when 'restart' keyword is invoked in COLUMBUS.resources file
    :param QMin:
    :param _iter: Iteration number
    :return:
    """
    QMin['newpath'] = str(os.path.join(QMin['savedir'], f'sp_{_iter - 1}'))
    if not checkpath(QMin['newpath'], aDIR=True):
        sys.exit(f"\n\nError: {QMin['newpath']} path not found. Unable to generate a restart")

    # Generate PCAP.RotMatOld.h5
    pathSAVE = QMin['savedir']
    eta_opt = float(QMin['eta_opt'])
    H0, W, ao_ovlp, projcap_object = get_capmat_H0_opencap(QMin)
    H_diag, Leigvc, Reigvc = ZO_TO_DIAG_energy(H0, W, eta_opt, corrected=False)

    RotMATnew = {}
    RotMATnew['Reigvc'] = Reigvc
    RotMATnew['Leigvc'] = Leigvc
    RotMATnew['ao_ovlp'] = ao_ovlp

    RotMATfile = os.path.join(pathSAVE, "PCAP.RotMatOld.h5")
    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=Reigvc)
        f.create_dataset("Leigvc", data=Leigvc)

    # generate_ad_den_files(QMin, RotMATnew)

    # Generate dets.old, mocoef.wfov.old, mocoef.old
    shutil.copy("%s/MOCOEFS/mocoef_mc.sp" % QMin['newpath'], "%s/mocoef.old" % pathSAVE)
    if QMin.get('track') != 'dens':
        shutil.copy("%s/TRACK/dets" % QMin['newpath'], "%s/dets.old" % pathSAVE)
        if QMin.get('track') == 'wfoverlap':
            shutil.copy("%s/MOCOEFS/mocoef_mc.sp" % QMin['newpath'], "%s/mocoef.wfov.old" % pathSAVE)

    elif QMin.get('track') == 'dens':
        addens_file = os.path.join(QMin['newpath'], 'TRACK', 'ADDENS.h5')
        with h5py.File(addens_file) as file:
            attach_mo = file['attach_mo'][:]
            detach_mo = file['detach_mo'][:]
        with h5py.File(RotMATfile, "a") as f:
            if "attach_mo" in f:
                del f["attach_mo"]
            f.create_dataset("attach_mo", data=attach_mo)
            if "detach_mo" in f:
                del f["detach_mo"]
            f.create_dataset("detach_mo", data=detach_mo)

    QMin['oldpath'] = copy.deepcopy(os.path.join(QMin['savedir'], f'sp_{_iter - 1}'))
    QMin['newpath'] = copy.deepcopy(os.path.join(QMin['savedir'], f'sp_{_iter}'))

    return


def cleanupWORK(QMin, path=None):
    pathDIR = QMin.get('oldpath')
    '''
    if not pathDIR:
        raise ValueError("Path not found in QMin['oldpath']")
    '''
    if path:
        pathTOcheck = [path]
    else:
        pathTOcheck = [
            os.path.join(pathDIR, 'ANION', 'WORK'),
            os.path.join(pathDIR, 'NEUTRAL', 'WORK'),
            os.path.join(pathDIR, 'WORK')
        ]

    for checkPath in pathTOcheck:
        if os.path.isdir(checkPath):
            for root, dirs, files in os.walk(checkPath):
                for file_name in files:
                    if file_name != 'daltaoin':
                        file_path = os.path.join(root, file_name)
                        try:
                            os.remove(file_path)
                            if 'debug' in QMin:
                                print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file {file_path}: {e}")
        else:
            print(f"Directory not found, skipping: {checkPath}")

    return


def columbusEngine(initial_positions, _iter, QMin):
    """
    Columbus-opencap wrapper interface to the optimizers that provides energy and gradient

    This can be integrated externally to any other optimizers as well (#TODO)
    :param initial_positions: position at each iteration
    :param _iter: iteration number
    :param QMin:
    :return: energy and gradient
    """
    pathDIR = QMin['pwd']
    os.makedirs(QMin['savedir'], exist_ok=True)
    QMin['newpath'] = str(os.path.join(QMin['savedir'], f'sp_{_iter}'))

    runerror = run_calc(initial_positions, _iter, QMin)

    if runerror == 0:
        cap_iwfmtpath = os.path.join(QMin['newpath'], 'ANION', 'WORK')
        cap_final_path = os.path.join(QMin['newpath'], 'CAP_INPS')
        os.makedirs(cap_final_path, exist_ok=True)
        generate_dens(QMin, cap_iwfmtpath)

        import glob
        capfiles = glob.glob(os.path.join(cap_iwfmtpath, "*iwfmt")) \
                   + [os.path.join(QMin['newpath'], 'MOLDEN', 'molden_mo_mc.sp')]

        if QMin['calculation'] == 'mcscf':
            capfiles += glob.glob(os.path.join(cap_iwfmtpath, "mcscfsm"))
        else:
            capfiles += glob.glob(os.path.join(cap_iwfmtpath, "tranls")) + glob.glob(
                os.path.join(cap_iwfmtpath, "ciudgsm"))

        for idx, cpfile in enumerate(capfiles):
            try:
                _basename = os.path.basename(cpfile)
                shutil.copy(cpfile, os.path.join(cap_final_path, _basename))
            except Exception as er:
                print(f"Failed to move {_basename}: {er}")
                traceback.print_exc()

        # copy neutral
        if QMin['calculation'] == 'mcscf':
            _neu_file = os.path.join(QMin['newpath'], 'NEUTRAL', 'WORK', 'mcscfsm')
        else:
            _neu_file = os.path.join(QMin['newpath'], 'NEUTRAL', 'WORK', 'ciudgsm')

        try:
            shutil.copy(_neu_file, os.path.join(cap_final_path, f'{os.path.basename(_neu_file)}_neutral'))
        except Exception as er:
            print(f"Failed to move {_neu_file}: {er}")
            traceback.print_exc()
    else:
        print("SEVERE error in COLUMBUS run: Runerror %s: " % runerror)
        sys.exit(911)

    gradients, gradients_CAP, ostring, eV = collectinfo(_iter, QMin)
    gradients_engine = []
    gradients_CAP_engine = []
    for _grad, _gradCAP in zip(gradients, gradients_CAP):
        gradients_engine.append([_grad['x'], _grad['y'], _grad['z']])
        gradients_CAP_engine.append([_gradCAP['x'], _gradCAP['y'], _gradCAP['z']])

    with open("%s/PCAP.out" % pathDIR, "a+") as f:
        f.write(ostring + "\n")

    if 'X' in QMin['atom_symbols']:
        QMin['atom_symbols'].remove('X')

    gradients_CAP_engine = np.array(gradients_CAP_engine)
    data = list(
        zip(QMin['atom_symbols'], gradients_CAP_engine[:, 0], gradients_CAP_engine[:, 1], gradients_CAP_engine[:, 2]))
    print_to_table("CAP gradients (Hartree/Bohr) of resonance state", ["Atom", "X", "Y", "Z"], data, align='right')

    QMin['oldpath'] = copy.deepcopy(QMin['newpath'])
    cleanupWORK(QMin)

    return eV, np.asarray(gradients_engine)


class CustomEngine(geometric.engine.Engine):
    '''A custom engine for geometric'''

    def __init__(self, molecule, QMin):
        self.mol = molecule
        self.rundir = QMin['pwd']
        super(CustomEngine, self).__init__(molecule)
        self._iter = 0
        if 'restart' in QMin:
            self._iter = QMin['iter']
            generate_restartfiles(QMin, self._iter)
        self.qmin = QMin

        # Initialize 'oldgeom' and 'newgeom' if not already in QMin
        self.qmin['oldgeom'] = self.qmin.get('oldgeom', None)
        self.qmin['newgeom'] = self.qmin.get('newgeom', None)

    def calc_new(self, coords, dirname):
        coords = coords.reshape(len(self.mol.elem), 3)
        coordsnew = [(e, xyz) for e, xyz in zip(self.mol.elem, coords)]

        self.qmin['coords'] = copy.deepcopy(coords)

        if self.qmin['newgeom'] is not None:
            self.qmin['oldgeom'] = copy.deepcopy(self.qmin['newgeom'])
        else:
            self.qmin['oldgeom'] = copy.deepcopy(coords)

        self.qmin['newgeom'] = copy.deepcopy(coords)

        starttime = datetime.datetime.now()

        print_to_file("\n\nCOLUMBUS Solver run: %i" % self._iter, self.rundir)
        data = list(zip(self.mol.elem, coords[:, 0], coords[:, 1], coords[:, 2]))
        print_to_table("Position (Bohr)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(self.rundir, 'iteration_details'), align='right')

        energy_complex, gradient = columbusEngine(coords, self._iter, self.qmin)
        energy = np.real(energy_complex)

        data = list(zip(self.qmin['act_state'], [energy_complex]))
        print_to_table("Projected CAP generated complex energies (Hartree)"
                       , ["Root", "Complex energies (Re, Im)"], data,
                       fname=os.path.join(self.rundir, 'iteration_details'))
        data = list(zip(self.mol.elem, gradient[:, 0], gradient[:, 1], gradient[:, 2]))
        print_to_table("Gradients (Hartree/Bohr)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(self.rundir, 'iteration_details'), align='right')

        endtime = datetime.datetime.now()
        print_to_file("\nElapsed time (hh:mm:ss) ==> %s\n\n" % (endtime - starttime), self.rundir)

        self._iter += 1
        self.qmin['iter'] = self._iter
        return {'energy': energy, 'gradient': gradient.ravel()}


def BernySolver(engine, QMin):
    '''Berny interface to columbus-opencap engine'''
    _iter = 0
    if 'restart' in QMin:
        _iter = QMin['iter']
        generate_restartfiles(QMin, _iter)

    _atoms, _lattice = yield
    pathRUN = str(QMin['pwd'])

    QMin['oldgeom'] = QMin.get('oldgeom', None)
    QMin['newgeom'] = QMin.get('newgeom', None)

    while True:
        coords = np.array([coord for _, coord in _atoms])
        elems = np.array([el for el, _ in _atoms])
        QMin['coords'] = copy.deepcopy(coords)

        if QMin['newgeom'] is not None:
            QMin['oldgeom'] = copy.deepcopy(QMin['newgeom'])
        else:
            QMin['oldgeom'] = copy.deepcopy(coords)

        QMin['newgeom'] = copy.deepcopy(coords)

        starttime = datetime.datetime.now()
        print_to_file("\n\nCOLUMBUS Solver run: %i" % _iter, pathRUN)

        data = list(zip(elems, coords[:, 0], coords[:, 1], coords[:, 2]))
        print_to_table("Position", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(pathRUN, 'iteration_details'), align='right')

        energy_complex, gradients = engine(coords, _iter, QMin)
        energy = np.real(energy_complex)

        data = list(zip(QMin['act_state'], [energy_complex]))
        print_to_table("Projected CAP generated complex energies (Hartree)"
                       , ["Root", "Complex energies (Re, Im)"], data,
                       fname=os.path.join(pathRUN, 'iteration_details'))
        data = list(zip(elems, gradients[:, 0], gradients[:, 1], gradients[:, 2]))
        print_to_table("Gradients (Hartree/Bohr)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(pathRUN, 'iteration_details'), align='right')

        endtime = datetime.datetime.now()
        print_to_file("\nElapsed time (hh:mm:ss) ==> %s\n\n" % (endtime - starttime), pathRUN)

        _iter += 1
        QMin['iter'] = _iter
        _atoms, _lattice = yield energy, gradients


def init_params(geomfile):
    """
    Parse geometry file and extract atomic symbols and coordinates.

    :param geomfile: str, path to the geometry file.
                     Expected format: Each line contains an atom symbol and its coordinates, e.g.,
                     H 0.0 0.0 0.0
    :return: tuple (geom, atom_symbols)
             - geom: np.ndarray of shape (N, 3), where N is the number of atoms, containing atomic coordinates.
             - atom_symbols: np.ndarray of shape (N,), containing atomic symbols as strings.
    """
    try:
        geomstrings = readfile(geomfile)
    except FileNotFoundError:
        raise ValueError(f"File '{geomfile}' not found.")

    geom = []
    atom_symbols = []

    for i, geom_line in enumerate(geomstrings):
        geom_line = geom_line.strip()
        if not geom_line:
            continue
        parts = geom_line.split()
        if parts[0] == 'X':
            continue
        if len(parts) < 5:
            raise ValueError(f"Invalid line format at line {i + 1}: '{geom_line}'")
        try:
            atom_symbols.append(parts[0])
            geom.append([float(coord) for coord in parts[2:5]])
        except ValueError:
            raise ValueError(f"Failed to parse coordinates at line {i + 1}: '{geom_line}'")

    geom = np.array(geom, dtype=float)
    atom_symbols = np.array(atom_symbols, dtype=str)

    return geom, atom_symbols


def checkpath(path, aFile=False, aDIR=True):
    """
    Check if a path exists, and if it is a file or directory based on the flags provided.

    Args:
        path (str): The path to check.
        aFile (bool): Set to True if the path should be a file.
        aDIR (bool): Set to True if the path should be a directory.

    Returns:
        bool: True if the path exists and matches the expected type, False otherwise.
    """
    if os.path.exists(path):
        if os.path.isdir(path) and aDIR:
            return True
        elif os.path.isfile(path) and aFile:
            return True
        else:
            print(f"The path {path} exists but is not the expected type.")
    else:
        print(f"The path {path} does not exist.")
    return False


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
                        optking_extras[key] = value
                    elif value.lower() == 'false':
                        value = False
                        optking_extras[key] = value
                    else:
                        try:
                            optking_extras[key] = float(value)
                        except:
                            optking_extras[key] = str(value)
                except ValueError:
                    print(f"Line format error: {line} in optking.params file!")
    return optking_extras


def optkingSolver(engine, QMin):
    '''Optking interface to columbus-opencap engine'''
    _iter = 0
    if 'restart' in QMin:
        _iter = QMin['iter']
        generate_restartfiles(QMin, _iter)

    pathRUN = str(QMin['pwd'])

    import optking
    import qcelemental as qcel

    atomic_numbers = [qcel.periodictable.to_Z(symbol) for symbol in QMin['atom_symbols']]
    geometry = np.column_stack((atomic_numbers, QMin['coords'] / ANG_TO_BOHR))
    mol = qcel.models.Molecule.from_data(geometry, dtype="numpy")
    conv_type = QMin.get('conv') or "gau_tight"  # gau_tight is the default choice
    optking_options = {"g_convergence": conv_type,
                       "opt_coordinates": "both"}

    try:
        optking_options_extra = read_optking_params(QMin)
        optking_options.update(optking_options_extra)
    except OSError as e:
        print('Reading of optking.params returned:\n\t%s' % e)

    opt = optking.CustomHelper(mol, optking_options)

    QMin['oldgeom'] = QMin.get('oldgeom', None)
    QMin['newgeom'] = QMin.get('newgeom', None)

    for step in range(_iter, QMin.get("geom_maxiter", 300)):

        if QMin['newgeom'] is not None:
            QMin['oldgeom'] = copy.deepcopy(QMin['newgeom'])
        else:
            QMin['oldgeom'] = copy.deepcopy(QMin['coords'])

        QMin['newgeom'] = copy.deepcopy(QMin['coords'])

        starttime = datetime.datetime.now()
        print_to_file("\n\nCOLUMBUS Solver run: %i" % _iter, pathRUN)

        data = list(zip(QMin['atom_symbols'], QMin['coords'][:, 0], QMin['coords'][:, 1], QMin['coords'][:, 2]))
        print_to_table("Position", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(pathRUN, 'iteration_details'), align='right')

        E, gX = engine(QMin['coords'], _iter, QMin)
        opt.E = np.real(E)
        opt.gX = gX

        _symmetrized_done = QMin.get('coords_symmetrized', False)
        if _symmetrized_done:
            opt.molsys.geom = copy.deepcopy(QMin['coords'])

        opt.compute()
        opt.take_step()
        conv = opt.test_convergence()

        # Update coords for the next run!
        QMin['coords'] = copy.deepcopy(opt.geom)

        data = list(zip(QMin['act_state'], [E]))
        print_to_table("Projected CAP generated complex energies (Hartree)"
                       , ["Root", "Complex energies (Re, Im)"], data,
                       fname=os.path.join(pathRUN, 'iteration_details'))
        data = list(zip(QMin['atom_symbols'], gX[:, 0], gX[:, 1], gX[:, 2]))
        print_to_table("Gradients (Hartree/Bohr)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(pathRUN, 'iteration_details'), align='right')

        endtime = datetime.datetime.now()
        print_to_file("\nElapsed time (hh:mm:ss) ==> %s\n\n" % (endtime - starttime), pathRUN)

        _iter += 1
        QMin['iter'] = _iter

        if conv is True:
            print("Optimization SUCCESS after %i iterations:" % (step + 1))
            break
    else:
        print("Optimization FAILURE (iteration: %i)\n" % (step + 1))

    return opt.geom


def _build_QMin(QMin):
    '''Reads template, resources file and builds QMin dictionary'''
    # Open and read the template file
    template_path = os.path.join(QMin['pwd'], 'COLUMBUS.template')
    template = readfile(template_path)
    resources_path = os.path.join(QMin['pwd'], 'COLUMBUS.resources')
    resources = readfile(resources_path)
    string = "\n\n====> COLUMBUS.resources/template variables:\n\n"

    for line in template:
        # Clean up the line and split it
        line = re.sub('#.*$', '', line).strip().lower().split()

        if not line:
            continue  # Skip empty lines

        key = line[0]
        value = line[1] if len(line) > 1 else None

        if key == 'nstates':
            QMin[key] = int(value)
        elif key == 'eta_opt':
            QMin[key] = float(value)
        elif key == 'ghost_atom':
            QMin[key] = []
        elif key == 'act_state':
            QMin[key] = [int(value)]
        elif key == 'cap_x':
            QMin[key] = float(value)
        elif key == 'cap_y':
            QMin[key] = float(value)
        elif key == 'cap_z':
            QMin[key] = float(value)
        elif key == 'calculation':
            QMin[key] = str(value)
        elif key == 'no_capgrad_correct':
            QMin[key] = []
        elif key == 'natorb':
            QMin[key] = []
        elif key == 'maxsqnorm':
            QMin[key] = float(value)
        elif key.lower() == 'symmetry':
            QMin[key] = True
        elif key.lower() == 'track':
            if value:
                QMin[key] = value
        elif key.lower() == 'screening':
            QMin[key] = float(value)

        if key in QMin:
            string += f"\t{key}: {QMin[key]}\n"

    for line in resources:
        # Clean up the line and split it
        line = re.sub('#.*$', '', line).strip().split()
        if not line:
            continue  # Skip empty lines

        key = line[0].lower()
        value = line[1] if len(line) > 1 else None

        if key == 'columbus':
            QMin[key] = os.path.abspath(value)
        elif key == 'scratchdir':
            QMin[key] = os.path.abspath(value)
        elif key == 'wfoverlap':
            QMin[key] = os.path.abspath(value)
        elif key == 'wfovdir':
            QMin[key] = os.path.abspath(value)
        elif key == 'memory':
            QMin[key] = int(value)
        elif key == 'ncpu':
            QMin[key] = int(value)
        elif key == 'savedir':
            QMin[key] = os.path.abspath(value)
        elif key == 'restart':
            QMin['iter'] = int(value)
            QMin[key] = []
        elif key == 'optimizer':
            QMin[key] = str(value).lower()
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
        elif key == 'inputdir':
            QMin[key] = os.path.abspath(value)

        if key in QMin:
            string += f"\t{key}: {QMin[key]}\n"

    # Verify required paths
    if 'columbus' not in QMin:
        QMin['columbus'] = os.path.abspath(os.getenv('COLUMBUS'))
        if not checkpath(QMin['columbus'], aDIR=True):
            sys.exit("Error: Columbus path not found.")

    if 'wfoverlap' in QMin:
        if not QMin.get('wfovdir'):
            _pathbin = os.path.normpath(QMin['wfoverlap'])
            _pathbin = os.path.dirname(os.path.dirname(_pathbin))
            QMin['wfovdir'] = os.path.join(_pathbin, 'wfoverlap')
            if not checkpath(QMin['wfovdir'], aDIR=True):
                sys.exit("Error: WFOverLap directory not found.")
            if not checkpath(QMin['wfoverlap'], aFile=True):
                sys.exit("Error: wfoverlap.x file not found.")

    for key in ['wfoverlap', 'wfovdir', 'columbus']:
        if key in QMin:
            string += f"\t{key}: {QMin[key]}\n"

    QMin['track'] = 'dens' if QMin.get('symmetry') else QMin.get('track', 'wfoverlap')
    string += f"\t{'track'}: {QMin['track']}\n"
    string_extra = f"\n\n*** Tracking is done by '{QMin.get('track')}' and symmetry is {str(QMin.get('symmetry'))}\n"
    if not QMin.get('symmetry'):
        string_extra += f"If symmetry should be on, cancel the job and choose following " \
                        f"keywords in \n{template_path}\n\n {'symmetry'}\n {'track '}{'dens'}/{'civecs'} (any one)\n\n"

    string += "\n"
    print_to_file(string, QMin['pwd'])
    print(string_extra)

    keys_needed = ['nstates', 'eta_opt', 'cap_x', 'cap_y', 'cap_z']
    for key in keys_needed:
        if not QMin.get(key):
            sys.exit(f'Specify {key} in MOLCAS.template')
    return QMin


class CustomFormatter(logging.Formatter):
    '''A custom logger for berny interface'''

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


QMin = {}


def main():
    global QMin

    # Initialize some variables
    columbus = os.environ.get('COLUMBUS')
    memory = 8000  # MB
    ncpu = 1
    # =============================

    QMin['pwd'] = str(sys.argv[1])
    print_to_file("\nSimulation submission path:\n %s" % QMin['pwd'], QMin['pwd'])
    QMin['columbus'] = columbus
    QMin['wfovdir'] = None
    QMin['memory'] = memory
    QMin['ncpu'] = ncpu
    QMin['calculation'] = []
    QMin['mcscf_guess'] = []
    QMin['natorb'] = []
    QMin['savedir'] = os.path.join(QMin['pwd'], 'SAVEDIR')
    QMin['iter'] = 0
    QMin['optimizer'] = 'optking'

    # Using default values here
    QMin['gradientmax'] = 4.50E-04
    QMin['gradientrms'] = 3.00E-04
    QMin['stepmax'] = 1.80E-03
    QMin['steprms'] = 1.20E-03
    QMin['deltae'] = 1.00e-05
    QMin['superweakdih'] = False
    QMin['symmetry'] = False

    # =============================
    # Read COLUMBUS.template/resources inputs
    QMin = _build_QMin(QMin)
    if QMin['symmetry']:
        if QMin.get('optimizer') != 'optking':
            QMin['optimizer'] = 'optking'
            print('\n\n*** Warning optimizer is set to optking when symmetry is on. \n\n')

    if 'restart' in QMin:
        geomfile = "%s/geom.restart" % QMin['pwd']
        print_to_file("\n\n This is a restart job reading: %s\n\n" % geomfile, QMin['pwd'])
    else:
        geomfile = "%s/geom" % QMin['pwd']

    initial_positions, atom_symbols = init_params(geomfile)
    QMin['coords'] = copy.deepcopy(initial_positions)
    QMin['natom'] = len(atom_symbols)
    molecule = geometric.molecule.Molecule()
    molecule.elem = [elem for elem in atom_symbols]
    molecule.xyzs = [initial_positions / ANG_TO_BOHR]
    QMin['atom_symbols'] = molecule.elem
    # =============================

    # Now exceute the external optimizers

    if QMin.get('optimizer') == 'berny':
        logger = logging.getLogger('Columbus_pyBerny_logger')
        if 'debug' in QMin:
            logger.setLevel(logging.DEBUG)
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
        berny.defaults = conv_params

        geom_init = geomlib.Geometry(species=atom_symbols, coords=initial_positions)
        optimizer = Berny(geom_init, logger=logger)

        # Optimize by pyberny
        solverBerny = BernySolver(columbusEngine, QMin)
        geom_final = optimize(optimizer, solverBerny, trajectory=None)

        # Print the final geometry
        geom_final = np.array([coord for _, coord in geom_final])

        data = list(zip(QMin['atom_symbols'], geom_final[:, 0] / ANG_TO_BOHR, geom_final[:, 1] / ANG_TO_BOHR,
                        geom_final[:, 2] / ANG_TO_BOHR))
        print_to_table("\n\nOptimized positions of the atoms (Angstrom)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(QMin['pwd'], 'iteration_details'), align='right')

    elif QMin.get('optimizer') == 'geometric':

        COLUMBUSengine = CustomEngine(molecule, QMin)
        conv_params = ['energy', QMin['deltae'],
                       'grms', QMin['gradientrms'],
                       'gmax', QMin['gradientmax'],
                       'drms', QMin['steprms'],
                       'dmax', QMin['stepmax']]

        outfile = os.path.join(QMin['pwd'], "geomeTRIC.log")
        with open(outfile, "w+") as _outfile:
            m = geometric.optimize.run_optimizer(customengine=COLUMBUSengine, converge=conv_params, check=1,
                                                 input=_outfile.name)

        data = list(zip(atom_symbols, m.xyzs[-1][:, 0], m.xyzs[-1][:, 1], m.xyzs[-1][:, 2]))
        print_to_table("\n\nOptimized positions of the atoms (Angstrom)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(QMin['pwd'], 'iteration_details'), align='right')

    elif QMin.get('optimizer') == 'optking':
        geom_final = optkingSolver(columbusEngine, QMin)
        data = list(zip(QMin['atom_symbols'], geom_final[:, 0] / ANG_TO_BOHR, geom_final[:, 1] / ANG_TO_BOHR,
                        geom_final[:, 2] / ANG_TO_BOHR))
        print_to_table("\n\nOptimized positions of the atoms (Angstrom)", ["Atom", "X", "Y", "Z"],
                       data, fname=os.path.join(QMin['pwd'], 'iteration_details'), align='right')


if __name__ == "__main__":
    main()
