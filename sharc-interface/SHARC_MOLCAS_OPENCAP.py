#!/usr/bin/env python3

# ******************************************
#
#    Modified Version of SHARC Program Suite
#
#    Based on SHARC (https://sharc-md.org/)
#    Originally Copyright (c) 2023 University of Vienna
#    Originally Copyright (c) 2023 University of Minnesota
#
#    This modified version has undergone substantial changes by
#    Soubhik Mondal at Boston University
#    and may no longer resemble the original SHARC software.
#
#    This file is distributed under the GNU General Public License v3 (GPLv3).
#    You may redistribute and modify it under the same license terms.
#
#    SHARC is licensed under GPL v3.
#    For license details, see <http://www.gnu.org/licenses/>.
#
# ******************************************

# Some functions in this file are directly derived from SHARC (with little to minor modifications)
# to maintain compatibility with SHARC (https://sharc-md.org/).
# These functions include:
'''
readfile, writefile, link, isbinary, eformat, measuretime, removekey, containsstring,
printheader, printQMin, printtasks, printcomplexmatrix, printgrad, printQMout,
makecmatrix, makermatrix, getversion, getcienergy, getsmate, getQMout, writeQMout,
writeQMoutsoc, writeQMoutgrad, writeQMoutnacsmat, writeQMouttime, writeQmoutPhases,
checkscratch, removequotes, getsh2caskey, get_sh2cas_environ, get_pairs, readQMin,
format_ci_vectors, get_determinants, decompose_csf, stripWORKDIR, moveJobIphs,
cleanupSCRATCH, get_zeroQMout
'''

# Some functions in this file are derived from SHARC and heavily modified, but their logic remains similar.
# These functions include:
'''
arrangeQMout, runjobs, run_calc, generate_joblist, runMOLCAS, setupWORKDIR,
writegeomfile, writeMOLCASinput, gettasks, readQMin
'''

# These parts remain under the GNU General Public License v3 (GPLv3).

# ======================================================================= #
# Modules:
# Operating system, isfile and related routines, move files, create directories
import os
import shutil
# External Calls to MOLCAS
import subprocess as sp
# Command line arguments
import sys
# Regular expressions
import re
# debug print for dicts and arrays
import pprint
# sqrt and other math
import math
# runtime measurement
import datetime
from copy import deepcopy
# parallel calculations
from multiprocessing import Pool
import concurrent.futures
import time
from socket import gethostname
import itertools
# write debug traces when in pool threads
import traceback
import numpy as np
import scipy.linalg as LA
import logging
import io
from contextlib import redirect_stdout
from tabulate import tabulate

opencap_driver = False
try:
    __import__("pyopencap")
    print("\nOpenCAP found and will be called in %s \n" % (sys.argv[0]))
    opencap_driver = True
except ImportError:
    print("Not found pyopencap")

#sys.path.append('./../utils/')
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
utils_dir = os.path.join(script_dir, "../utils")
sys.path.append(utils_dir)

import copy
import ParseFile_MOLCAS
import h5py
from functools import reduce

# ======================================================================= #

version = '3.0.OPENCAP.1.2.x'
versiondate = datetime.date(2025, 1, 30)

# ======================================================================= #
# holds the system time when the script was started
starttime = datetime.datetime.now()

# global variables for printing (PRINT gives formatted output, DEBUG gives raw output)
DEBUG = False
PRINT = True

# hash table for conversion of multiplicity to the keywords used in MOLCAS
IToMult = {
    1: 'Singlet',
    2: 'Doublet',
    3: 'Triplet',
    4: 'Quartet',
    5: 'Quintet',
    6: 'Sextet',
    7: 'Septet',
    8: 'Octet',
    'Singlet': 1,
    'Doublet': 2,
    'Triplet': 3,
    'Quartet': 4,
    'Quintet': 5,
    'Sextet': 6,
    'Septet': 7,
    'Octet': 8
}

# hash table for conversion of polarisations to the keywords used in MOLCAS
IToPol = {
    0: 'X',
    1: 'Y',
    2: 'Z',
    'X': 0,
    'Y': 1,
    'Z': 2
}

# conversion factors
au2a = 0.529177211
rcm_to_Eh = 4.556335e-6


# =============================================================================================== #
# =============================================================================================== #
# =========================================== general routines ================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out


# ======================================================================= #


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


# ======================================================================= #


def link(PATH, NAME, crucial=True, force=True):
    # do not create broken links
    if not os.path.exists(PATH):
        print('Source %s does not exist, cannot create link!' % (PATH))
        sys.exit(14)
    if os.path.islink(NAME):
        if not os.path.exists(NAME):
            # NAME is a broken link, remove it so that a new link can be made
            os.remove(NAME)
        else:
            # NAME is a symlink pointing to a valid file
            if force:
                # remove the link if forced to
                os.remove(NAME)
            else:
                print('%s exists, cannot create a link of the same name!' % (NAME))
                if crucial:
                    sys.exit(15)
                else:
                    return
    elif os.path.exists(NAME):
        # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
        print('%s exists, cannot create a link of the same name!' % (NAME))
        if crucial:
            sys.exit(16)
        else:
            return
    os.symlink(PATH, NAME)


# ======================================================================= #


def isbinary(path):
    return (re.search(r':.* text', sp.Popen(["file", '-L', path], stdout=sp.PIPE).stdout.read()) is None)


# ======================================================================= #


def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))


# ======================================================================= #

def time_string(starttime, endtime):
    runtime = endtime - starttime
    if PRINT or DEBUG:
        hours = runtime.seconds / 3600
        minutes = runtime.seconds / 60 - hours * 60
        seconds = runtime.seconds % 60
        # print ('==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days,hours,minutes,seconds))
    total_seconds = runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds / 1.e6
    return total_seconds


def measuretime():
    '''Calculates the time difference between global variable starttime and the time of the call of measuretime.

    Prints the Runtime, if PRINT or DEBUG are enabled.

    Arguments:
    none

    Returns:
    1 float: runtime in seconds'''

    endtime = datetime.datetime.now()
    runtime = endtime - starttime
    if PRINT or DEBUG:
        hours = runtime.seconds // 3600
        minutes = runtime.seconds // 60 - hours * 60
        seconds = runtime.seconds % 60
        print('==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days, hours, minutes, seconds))
    total_seconds = runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds // 1.e6
    return total_seconds


# ======================================================================= #


def removekey(d, key):
    '''Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary'''

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d


# ======================================================================= #         OK


def containsstring(string, line):
    '''Takes a string (regular expression) and another string. Returns True if the first string is contained in the second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean'''

    a = re.search(string, line)
    if a:
        return True
    else:
        return False


# =============================================================================================== #
# =============================================================================================== #
# ============================= iterator routines  ============================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def itmult(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return


# ======================================================================= #


def itnmstates(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return


# =============================================================================================== #
# =============================================================================================== #
# =========================================== print routines ==================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #


def printheader():
    # This function is adapted from SHARC (https://sharc-md.org/)
    # Modifications by Soubhik Mondal to include OPENCAP and update authorship details.

    print(starttime, gethostname(), os.getcwd())
    if not PRINT:
        return
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '||' + ' ' * 20 + 'SHARC - OPENCAP (with MOLCAS) - Interface' + ' ' * 19 + '||\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '||' + ' ' * 20 + 'Authors: Soubhik Mondal, Ksenia Bravaya ' + ' ' * 20 + '||\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '||' + ' ' * (36 - (len(version) + 1) // 2) + 'Version: %s' % (version) + ' ' * (
            35 - (len(version)) // 2) + '||\n'
    lens = len(versiondate.strftime("%d.%m.%y"))
    string += '||' + ' ' * (37 - lens // 2) + 'Date: %s' % (versiondate.strftime("%d.%m.%y")) + ' ' * (
            37 - (lens + 1) // 2) + '||\n'
    string += '||' + ' ' * 80 + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

# ======================================================================= #


def printQMin(QMin):
    if DEBUG:
        pprint.pprint(QMin)
    if not PRINT:
        return
    print('==> QMin Job description for:\n%s' % (QMin['comment']))

    string = 'Tasks:  '
    if 'h' in QMin:
        string += '\tH'
    if 'soc' in QMin:
        string += '\tSOC'
    if 'dm' in QMin:
        string += '\tDM'
    if 'grad' in QMin:
        string += '\tGrad'
    if 'nacdr' in QMin:
        string += '\tNac(ddr)'
    if 'nacdt' in QMin:
        string += '\tNac(ddt)'
    if 'overlap' in QMin:
        string += '\tOverlaps'
    if 'angular' in QMin:
        string += '\tAngular'
    if 'ion' in QMin:
        string += '\tDyson norms'
    if 'dmdr' in QMin:
        string += '\tDM-Grad'
    if 'socdr' in QMin:
        string += '\tSOC-Grad'
    if 'phases' in QMin:
        string += '\tPhases'
    print(string)

    string = 'States: '
    for i in itmult(QMin['states']):
        string += '\t%i %s' % (QMin['states'][i - 1], IToMult[i])
    print(string)

    string = 'Method: \t'
    if 'pcap' in QMin:
        string += 'Projected CAP-SA(%i' % int(QMin['template']['roots'][0])
    else:
        string += 'SA(%i' % (QMin['template']['roots'][0])
    for i in QMin['template']['roots'][1:]:
        string += '|%i' % (i)
    string += ')-'
    string += QMin['template']['method'].upper()
    string += '(%i,%i)/%s' % (QMin['template']['nactel'], QMin['template']['ras2'], QMin['template']['basis'])
    parts = []
    if QMin['template']['cholesky']:
        parts.append('RICD')
    if not QMin['template']['no-douglas-kroll']:
        parts.append('Douglas-Kroll')
    if QMin['method'] > 0 and QMin['template']['ipea'] != 0.25:
        parts.append('IPEA=%4.2f' % (QMin['template']['ipea']))
    if QMin['method'] > 0 and QMin['template']['imaginary'] != 0.00:
        parts.append('Imaginary Shift=%4.2f' % (QMin['template']['imaginary']))
    if QMin['template']['frozen'] != -1:
        parts.append('CASPT2 frozen orbitals=%i' % (QMin['template']['frozen']))
    if len(parts) > 0:
        string += '\t('
        string += ','.join(parts)
        string += ')'
    print(string)
    # say, if CAS(n-1,m) is used for any multiplicity
    oddmults = False
    for i in QMin['statemap'].values():
        if (QMin['template']['nactel'] + i[0]) % 2 == 0:
            oddmults = True
    if oddmults:
        string = '\t\t' + ['Even ', 'Odd '][QMin['template']['nactel'] % 2 == 0]
        string += 'numbers of electrons are treated with CAS(%i,%i).' % (
            QMin['template']['nactel'] - 1, QMin['template']['ras2'])
        print(string)

    string = 'Found Geo'
    if 'veloc' in QMin:
        string += ' and Veloc! '
    else:
        string += '! '
    string += 'NAtom is %i.\n' % (QMin['natom'])
    print(string)

    string = '\nGeometry in Bohrs:\n'
    for i in range(QMin['natom']):
        string += '%s ' % (QMin['geo'][i][0])
        for j in range(3):
            string += '% 7.4f ' % (QMin['geo'][i][j + 1])
        string += '\n'
    print(string)

    if 'veloc' in QMin:
        string = ''
        for i in range(QMin['natom']):
            string += '%s ' % (QMin['geo'][i][0])
            for j in range(3):
                string += '% 7.4f ' % (QMin['veloc'][i][j])
            string += '\n'
        print(string)

    if 'grad' in QMin:
        string = 'Gradients:   '
        for i in range(1, QMin['nmstates'] + 1):
            if i in QMin['grad']:
                string += 'X '
            else:
                string += '. '
        string += '\n'
        print(string)

    if 'nacdr' in QMin:
        string = 'Non-adiabatic couplings:\n'
        for i in range(1, QMin['nmstates'] + 1):
            for j in range(1, QMin['nmstates'] + 1):
                if [i, j] in QMin['nacdr'] or [j, i] in QMin['nacdr']:
                    string += 'X '
                else:
                    string += '. '
            string += '\n'
        print(string)

    if 'overlap' in QMin:
        string = 'Overlaps:\n'
        for i in range(1, QMin['nmstates'] + 1):
            for j in range(1, QMin['nmstates'] + 1):
                if [i, j] in QMin['overlap'] or [j, i] in QMin['overlap']:
                    string += 'X '
                else:
                    string += '. '
            string += '\n'
        print(string)

    for i in QMin:
        if not any([i == j for j in
                    ['h', 'dm', 'soc', 'dmdr', 'socdr', 'geo', 'veloc', 'states', 'comment', 'LD_LIBRARY_PATH', 'grad',
                     'nacdr', 'ion', 'overlap', 'template']]):
            if not any([i == j for j in ['ionlist', 'ionmap']]) or DEBUG:
                string = i + ': '
                string += str(QMin[i])
                print(string)
    print('\n')
    sys.stdout.flush()


# ======================================================================= #


def printtasks(tasks):
    '''If PRINT, prints a formatted table of the tasks in the tasks list.

    Arguments:
    1 list of lists: tasks list (see gettasks for specs)'''

    # if DEBUG:
    # pprint.pprint(tasks)
    if not PRINT:
        return
    print('==> Task Queue:\n')
    for i in range(len(tasks)):
        task = tasks[i]
        if task[0] == 'gateway':
            print('GATEWAY')
        elif task[0] == 'seward':
            print('SEWARD')
        elif task[0] == 'link':
            print('Link\t%s\t--> \t%s' % (task[2], task[1]))
        elif task[0] == 'copy':
            print('Copy\t%s\t==> \t%s' % (task[1], task[2]))
        elif task[0] == 'rasscf':
            print('RASSCF\tMultiplicity: %i\tStates: %i\tJOBIPH=%s\tLUMORB=%s' % (task[1], task[2], task[3], task[4]))
        elif task[0] == 'rasscf-cms':
            print('RASSCF-CMS\tMultiplicity: %i\tStates: %i\tJOBIPH=%s\tLUMORB=%s\tRLXROOT=%i' % (
                task[1], task[2], task[3], task[4], task[5]))
        elif task[0] == 'alaska':
            print('ALASKA')
        elif task[0] == 'mclr':
            print('MCLR')
        elif task[0] == 'mclr-cms':
            print('MCLR-CMS')
        elif task[0] == 'caspt2':
            print('CASPT2\tMultiplicity: %i\tStates: %i\tMULTISTATE=%s' % (task[1], task[2], task[3]))
        elif task[0] == 'cms-pdft':
            print('CMS-PDFT\tFunctional: %s' % (task[1]))
        elif task[0] == 'rassi':
            print('RASSI\t%s\tStates: %s' % (
                {'soc': 'Spin-Orbit Coupling', 'dm': 'Dipole Moments', 'overlap': 'Overlaps', 'TRD1': 'TDM/SDM'}[
                    task[1]],
                task[2]))
        else:
            print(task)
    print('\n')
    sys.stdout.flush()


# ======================================================================= #


def printcomplexmatrix(matrix, states):
    '''Prints a formatted matrix. Zero elements are not printed, blocks of different mult and MS are delimited by dashes. Also prints a matrix with the imaginary parts, of any one element has non-zero imaginary part.

    Arguments:
    1 list of list of complex: the matrix
    2 list of integers: states specs'''

    nmstates = 0
    for i in range(len(states)):
        nmstates += states[i] * (i + 1)
    string = 'Real Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].real == 0.:
                string += ' ' * 11
            else:
                string += '% .3e ' % (matrix[istate][jstate].real)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    print(string)
    imag = False
    string = 'Imaginary Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].imag == 0.:
                string += ' ' * 11
            else:
                imag = True
                string += '% .3e ' % (matrix[istate][jstate].imag)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    string += '\n'
    if imag:
        print(string)


# ======================================================================= #


def printgrad(grad, natom, geo):
    '''Prints a gradient or nac vector. Also prints the atom elements. If the gradient is identical zero, just prints one line.

    Arguments:
    1 list of list of float: gradient
    2 integer: natom
    3 list of list: geometry specs'''

    string = ''
    iszero = True
    for atom in range(natom):
        string += '%i\t%s\t' % (atom + 1, geo[atom][0])
        for xyz in range(3):
            if grad[atom][xyz] != 0:
                iszero = False
            g = grad[atom][xyz]
            if isinstance(g, float):
                string += '% .5f\t' % (g)
            elif isinstance(g, complex):
                string += '% .5f\t% .5f\t\t' % (g.real, g.imag)
        string += '\n'
    if iszero:
        print('\t\t...is identical zero...\n')
    else:
        print(string)


def print_complex_matrix(matrix, header=None, formatting="{real:14.9f}{imag:14.9f}"):
    '''
    Prints a complex matrix in SHARC (AIMD driver printouts) format.
    :param matrix:
    :param header:
    :param formatting:
    :return: string
    '''
    n = matrix.shape[0]
    try:
        m = matrix.shape[1]
    except IndexError:
        m = 1
    if header is not None:
        print(header)
    for i in range(n):
        for j in range(m):
            try:
                value = formatting.format(real=matrix[i, j].real, imag=matrix[i, j].imag)
            except IndexError:
                value = formatting.format(real=matrix[i].real, imag=matrix[i].imag)
            print(value, end='')
            if j != m - 1:
                print(' ', end='')
            else:
                print("")
    print('\n')


# ======================================================================= #
def printQMout(QMin, QMout):
    '''If PRINT, prints a summary of all requested QM output values. Matrices are formatted using printcomplexmatrix, vectors using printgrad.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout'''

    if not PRINT:
        return
    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    print('===> Results:\n')
    # Hamiltonian matrix, real or complex
    if 'h' in QMin or 'soc' in QMin:
        eshift = math.ceil(QMout['h'][0][0].real)
        print('=> Hamiltonian Matrix:\nDiagonal Shift: %9.2f' % (eshift))
        matrix = deepcopy(QMout['h'])
        for i in range(nmstates):
            matrix[i][i] -= eshift
        printcomplexmatrix(matrix, states)
    # Dipole moment matrices
    if 'dm' in QMin:
        print('=> Dipole Moment Matrices:\n')
        for xyz in range(3):
            print('Polarisation %s:' % (IToPol[xyz]))
            matrix = QMout['dm'][xyz]
            printcomplexmatrix(matrix, states)
    # Gradients
    if 'grad' in QMin:
        print('=> Gradient Vectors:\n')
        istate = 0
        for imult, i, ms in itnmstates(states):
            print('%s\t%i\tMs= % .1f:' % (IToMult[imult], i, ms))
            printgrad(QMout['grad'][istate], natom, QMin['geo'])
            istate += 1
    # Nonadiabatic couplings
    if 'nacdr' in QMin:
        print('=> Analytical Non-adiabatic coupling vectors:\n')
        istate = 0
        for imult, i, msi in itnmstates(states):
            jstate = 0
            for jmult, j, msj in itnmstates(states):
                if imult == jmult and msi == msj:
                    print('%s\tStates %i - %i\tMs= % .1f:' % (IToMult[imult], i, j, msi))
                    printgrad(QMout['nacdr'][istate][jstate], natom, QMin['geo'])
                jstate += 1
            istate += 1
    # Overlaps
    if 'overlap' in QMin:
        print('=> Overlap matrix:\n')
        matrix = QMout['overlap']
        printcomplexmatrix(matrix, states)
        if 'phases' in QMout:
            print('=> Wavefunction Phases:\n')
            for i in range(nmstates):
                print('% 3.1f % 3.1f' % (QMout['phases'][i].real, QMout['phases'][i].imag))
            print('\n')
    sys.stdout.flush()


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Matrix initialization ================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #         OK
def makecmatrix(a, b):
    '''Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex'''

    mat = [[complex(0., 0.) for i in range(a)] for j in range(b)]
    return mat


# ======================================================================= #         OK


def makermatrix(a, b):
    '''Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real'''

    mat = [[0. for i in range(a)] for j in range(b)]
    return mat


# =============================================================================================== #
# =============================================================================================== #
# =========================================== output extraction ================================= #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def getversion(out, MOLCAS):
    allowedrange = [(18.0, 25.999), (8.29999, 9.30001)]
    # first try to find $MOLCAS/.molcasversion
    molcasversion = os.path.join(MOLCAS, '.molcasversion')
    if os.path.isfile(molcasversion):
        vf = open(molcasversion)
        string = vf.readline()
        vf.close()
        print('Content: "%s"\n' % string)
    # otherwise try to read this from the output file
    else:
        string = ''
        for i in range(50):
            line = out[i]
            s = line.split()
            for j, el in enumerate(s):
                if 'version' in el:
                    string = s[j + 1]
                    break
            if string != '':
                break
    a = re.search('[0-9]+\\.[0-9]+', string)
    if a is None:
        print(
            'No MOLCAS version found.\nCheck whether MOLCAS path is set correctly in MOLCAS.resources\nand whether $MOLCAS/.molcasversion exists.')
        sys.exit(17)
    v = float(a.group())
    if not any([i[0] <= v <= i[1] for i in allowedrange]):
        # allowedrange[0]<=v<=allowedrange[1]:
        print('MOLCAS version %3.1f not supported! ' % (v))
        sys.exit(18)
    if DEBUG:
        print('Found MOLCAS version %3.1f\n' % (v))
    return v


# ======================================================================= #
# =====                                                         ========== #
#           OpenCAP call functions and utility functions                  #
# =====                                                         ========== #
# ======================================================================= #
def _biorthogonalize(Leigvc, Reigvc):
    '''
    Does biorthogonalization via LU decomposition
    given a right and left eigen vectors
    :param Leigvc:
    :param Reigvc:
    :return: Leigvc, Reigvc
    '''
    M = Leigvc.T @ Reigvc
    P, L, U = LA.lu(M)
    Linv = LA.inv(L)
    Uinv = LA.inv(U)
    Leigvc = np.dot(Linv, Leigvc.T)
    Reigvc = np.dot(Reigvc, Uinv)
    Leigvc = Leigvc.T
    return Leigvc, Reigvc


def _biorthogonalize_w_degenblocks(Leigvc, Reigvc, energies):
    '''
        Adapted from opencap-pyscf interface
    '''
    degen_thresh = 1E-5
    degen_blocks = []
    cur_block = [0]
    cur_energy = energies[0]
    final_Leigvc = np.zeros(Leigvc.shape, dtype=complex)
    final_Reigvc = np.zeros(Reigvc.shape, dtype=complex)
    for i in range(1, len(energies)):
        if np.abs(energies[i] - cur_energy) < degen_thresh:
            cur_block.append(i)
        else:
            degen_blocks.append(cur_block.copy())
            cur_block = [i]
            cur_energy = energies[i]
    degen_blocks.append(cur_block.copy())
    for block in degen_blocks:
        if len(block) == 1:
            index = block[0]
            final_Reigvc[:, index] = Reigvc[:, index]
            final_Leigvc[:, index] = Leigvc[:, index]
        else:
            temp_Leigvc = np.zeros((Leigvc.shape[0], len(block)), dtype=complex)
            temp_Reigvc = np.zeros((Reigvc.shape[0], len(block)), dtype=complex)
            for i, index in enumerate(block):
                temp_Leigvc[:, i] = Leigvc[:, index]
                temp_Reigvc[:, i] = Reigvc[:, index]
            temp_Leigvc, temp_Reigvc = _biorthogonalize_reordered(temp_Leigvc, temp_Reigvc)
            for i, index in enumerate(block):
                final_Reigvc[:, index] = temp_Reigvc[:, i]
                final_Leigvc[:, index] = temp_Leigvc[:, i]
    return _biorthogonalize_reordered(final_Leigvc, final_Reigvc)


def _biorthogonalize_reordered(Leigvc, Reigvc):
    '''
        Adapted from opencap-pyscf interface
    '''
    M = Leigvc.T @ Reigvc
    P, L, U = LA.lu(M)
    Linv = LA.inv(L)
    Uinv = LA.inv(U)
    Leigvc = np.dot(Linv, Leigvc.T)
    Reigvc = np.dot(Reigvc, Uinv)
    Leigvc = Leigvc.T
    M = Leigvc.T @ Reigvc
    L_reordered = np.zeros(Leigvc.shape, dtype=complex)
    for i in range(0, len(M)):
        jmax = np.argmax(Leigvc.T @ Reigvc[:, i])
        L_reordered[:, i] = Leigvc[:, jmax]
    return L_reordered, Reigvc


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


def LUdiagonalize(A):
    '''
    Creates Left and right eigenvectors for a non-hermitian matrix A

    eigen values are sorted according to the real part.

    (Uses biorthogonalization)

    :param A:
    :return: Left and right eigen vectors and sorted eigen values
    '''

    ReVs, ReVc = _sort_eigenvectors(*LA.eig(A))
    LeVs, LeVc = _sort_eigenvectors(*LA.eig(A.T))

    #    LeVc, ReVc = _biorthogonalize(LeVc, ReVc)
    LeVc, ReVc = _biorthogonalize_w_degenblocks(LeVc, ReVc, ReVs)

    _overlap = LeVc.T @ ReVc
    _orthogonal = np.allclose(_overlap, np.eye(np.shape(A)[-1]))
    if not _orthogonal:
        print("Warning: Left and right eigenvectors are not orthonormal!")
    return LeVc, ReVc, np.diag(LeVs)


def diagonalize_cmatrix(A):
    '''
    Creates Left and right eigenvectors for a non-hermitian matrix A

    Eigen values are sorted according to the real part.

    left-eigen-vector = np.transpose(right-eigen-vector)

    :param A:
    :return: Left and right eigen vectors and sorted eigen values
    '''

    eigv, Reigvc = _sort_eigenvectors(*LA.eig(A))
    '''
    for istate in range(len(eigv)):
        Reigvc[:, istate] /= np.sqrt(sum(i * i for i in Reigvc[:, istate]))
    '''
    W = reduce(np.dot, (Reigvc.T, Reigvc))
    W_sqrt = LA.sqrtm(W)
    W_inv_sqrt = LA.inv(W_sqrt)
    Reigvc = reduce(np.dot, (Reigvc, W_inv_sqrt))

    return Reigvc, Reigvc, np.diag(eigv)


def _opencap_dicts(out, QMin, do_numerical=None):
    fnamex = os.path.splitext(out)[0]
    RASSI_FILE = "%s.rassi.h5" % fnamex
    OUTPUT_FILE = "%s.out" % fnamex

    if 'xms_correction' in QMin:
        method = "xms-caspt2"
        RASSI_FILE = os.path.join(QMin['scratchdir'], 'TRD1', "MOLCAS.rassi.h5")
        OUTPUT_FILE = os.path.join(QMin['scratchdir'], 'xms_corr', "MOLCAS.out")
    else:
        method = "sa-casscf"

    sys_dict = {"molecule": "molcas_rassi", "basis_file": RASSI_FILE}

    es_dict = {"package": "openmolcas",
               "method": method,
               "molcas_output": OUTPUT_FILE,
               "rassi_h5": RASSI_FILE}

    if QMin.get('cap_type') == 'voronoi':
        cap_dict = {"cap_type": "voronoi",
                    "r_cut": f"{QMin['template']['r_cut']}",
                    "radial_precision": "16",
                    "angular_points": "590"}
    else:
        cap_dict = {"cap_type": "box",
                    "cap_x": f"{QMin['template']['cap_x']}",
                    "cap_y": f"{QMin['template']['cap_y']}",
                    "cap_z": f"{QMin['template']['cap_z']}",
                    "Radial_precision": "16",
                    "angular_points": "590"}
    if do_numerical:
        cap_dict["do_numerical"] = "true"

    return sys_dict, es_dict, cap_dict


def CAPG_MAT(outfilename, nroots, QMin, do_numerical=None, saveRASSI=True):
    '''
    :param outfilename: molcas output file with .out extension.
    :param nroots: Number of states in state averaging.
    :return: box-CAP matrix projected on analytical derivative of Gaussian basis function.
             augmented by other CAP contribution to derivative.
    '''
    if opencap_driver:
        os.environ['OMP_NUM_THREADS'] = str(QMin['ncpu_avail'])
        #        os.environ['MKL_NUM_THREADS'] = str(QMin['ncpu_avail'])
        import pyopencap

    '''
    I am gonna wrap the outputs from OPENCAP-MD
    '''
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])
    ostr = io.StringIO()
    _start = datetime.datetime.now()

    sys.stdout.write("\n|%s %s %s|\n" % ("=" * 20, "OpenCAP run details:", "=" * 20))
    sys.stdout.write("\nCalculating gradient contribution from CAP via OpenCAP driver. \n")
    sys.stdout.flush()
    with redirect_stdout(ostr):
        sys_dict, es_dict, cap_dict = _opencap_dicts(outfilename, QMin, do_numerical)
        s = pyopencap.System(sys_dict)
        pc = pyopencap.CAP(s, cap_dict, nroots)
        pc.read_data(es_dict)
        pc.compute_projected_capG()
        _WG = pc.get_projected_capG()
        # Derivative of the CAP matrix with respect to directions
        pc.compute_projected_cap_der()
        _WD = pc.get_projected_cap_der()

    filename = os.path.join(QMin['scratchdir'], "master/MOLCAS.rasscf.h5")
    with h5py.File(filename, 'r') as file:
        _nuc_charges = [i for i in file['CENTER_CHARGES']]
    if DEBUG:
        print('HDF5 File Read:\t==>\t%s' % (filename))

    # SBK: we need to get rid of Ghost atom data
    _WG_Xfree = []
    _WD_Xfree = []
    for ct, _data in enumerate(_WG):
        if _nuc_charges[ct] != 0.0:
            _WG_Xfree.append(_data)
            _WD_Xfree.append(_WD[ct])

    # Collect all gradient contributions.
    Grad_correct = {}
    for iatom in range(QMin['natom']):
        Grad_correct[iatom] = {'x': _WG_Xfree[iatom]['x'] + _WD_Xfree[iatom]['x'],
                               'y': _WG_Xfree[iatom]['y'] + _WD_Xfree[iatom]['y'],
                               'z': _WG_Xfree[iatom]['z'] + _WD_Xfree[iatom]['z']}

    sys.stdout.write(f"{log_stream.getvalue()}")
    sys.stdout.write(f"{ostr.getvalue()}")
    _end = datetime.datetime.now()
    sys.stdout.write(
        "\n|%s %s %s|\n" % ("=" * 20, "End of OpenCAP run (%s Seconds)." % time_string(_start, _end), "=" * 20))
    sys.stdout.flush()

    if not saveRASSI:
        pathDEL = os.path.join(QMin['scratchdir'], 'TRD1/')
        if os.path.exists(pathDEL):
            shutil.rmtree(pathDEL)
        else:
            print("\nWarning: Path %s does not exist, nothing to remove.\n" % pathDEL)

    return Grad_correct


def parse_MOLCAS_Hamiltonian(outfilename, QMin, do_numerical=None, saveRASSI=True):
    nroots = QMin['template']['roots'][0]
    _H_CAP = np.zeros([nroots, nroots], dtype=complex)

    if opencap_driver:
        os.environ['OMP_NUM_THREADS'] = str(QMin['ncpu_avail'])
        #        os.environ['MKL_NUM_THREADS'] = str(QMin['ncpu_avail'])
        import pyopencap
        from pyopencap.analysis import CAPHamiltonian

    try:
        if opencap_driver:
            log_stream = io.StringIO()
            log_handler = logging.StreamHandler(log_stream)
            logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])
            _start = datetime.datetime.now()

            ostr = io.StringIO()

            sys.stdout.write("\n|%s %s %s|\n" % ("=" * 20, "OpenCAP run details:", "=" * 20))
            sys.stdout.write("\nCalculating CAP matrix via OpenCAP driver. \n")
            sys.stdout.flush()

            def _box_cap(x, y, z, w):
                cap_values = []
                cap_x0 = QMin['template']['cap_x']
                cap_y0 = QMin['template']['cap_y']
                cap_z0 = QMin['template']['cap_z']

                for i in range(0, len(x)):
                    result = 0.0
                    if abs(x[i]) > cap_x0:
                        result += (abs(x[i]) - cap_x0) ** 2.0
                    if abs(y[i]) > cap_y0:
                        result += (abs(y[i]) - cap_y0) ** 2.0
                    if abs(z[i]) > cap_z0:
                        result += (abs(z[i]) - cap_z0) ** 2.0
                    result = w[i] * result
                    cap_values.append(result)
                return cap_values

            with redirect_stdout(ostr):
                sys_dict, es_dict, cap_dict = _opencap_dicts(outfilename, QMin, do_numerical)
                s = pyopencap.System(sys_dict)
                if QMin.get('cap_type') == 'box':
                    pc = pyopencap.CAP(s, cap_dict, nroots, _box_cap)
                else:
                    pc = pyopencap.CAP(s, cap_dict, nroots)
                pc.read_data(es_dict)
                pc.compute_projected_cap()
                H_CAP = pc.get_projected_cap()
                # Symmetrize the CAP, why? Because I can do it!
                H_CAP = 0.5 * (H_CAP + H_CAP.T)
                H0 = pc.get_H()

            sys.stdout.write(f"{log_stream.getvalue()}")
            sys.stdout.write(f"{ostr.getvalue()}")
            _end = datetime.datetime.now()
            sys.stdout.write(
                "\n|%s %s %s|\n" % ("=" * 20, "End of OpenCAP run (%s Seconds)." % time_string(_start, _end), "=" * 20))
            sys.stdout.flush()
    except Exception as e:
        print("\nWarning: OpenCAP driver not configured correctly. CAP matrix is zero.\n")
        traceback.print_exc()

    CAPH = CAPHamiltonian(H0=H0, W=H_CAP)
    pathSAVE = os.path.join(QMin['scratchdir'], 'master')

    CAPH.export("%s/CAPMAT.out" % pathSAVE)
    if not saveRASSI:
        pathDEL = os.path.join(QMin['scratchdir'], 'TRD1')
        if os.path.exists(pathDEL):
            shutil.rmtree(pathDEL)
        else:
            print("\nWarning: Path %s does not exist, nothing to remove.\n" % pathDEL)

    return H0, 1.0j * H_CAP, pc, s


# ======================================================================= #
def getcienergy(out, mult, state, version, method, dkh):
    '''Searches a complete MOLCAS output file for the MRCI energy of (mult,state).

    Arguments:
    1 list of strings: MOLCAS output
    2 integer: mult
    3 integer: state

    Returns:
    1 float: total CI energy of specified state in hartree'''

    if method == 0:
        modulestring = '&RASSCF'
        spinstring = 'Spin quantum number'
        if dkh:
            energystring = 'Final state energy(ies)'
        else:
            energystring = '::    RASSCF root number'
            stateindex = 4
            enindex = 7
    elif method == 1:
        modulestring = '&CASPT2'
        spinstring = 'Spin quantum number'
        energystring = '::    CASPT2 Root'
        stateindex = 3
        enindex = 6
    elif method == 2:
        modulestring = '&CASPT2'
        spinstring = 'Spin quantum number'
        energystring = '::    MS-CASPT2 Root'
        stateindex = 3
        enindex = 6
    elif method == 3:
        modulestring = '&MCPDFT'
        spinstring = 'Spin quantum number'
        energystring = 'Total MC-PDFT energy for state'
        stateindex = 5
        enindex = 6
    elif method == 4:
        modulestring = '&MCPDFT'
        spinstring = 'Spin quantum number'
        energystring = '::    XMS-PDFT Root'
        stateindex = 3
        enindex = 6
    elif method == 5:
        modulestring = '&MCPDFT'
        spinstring = 'Spin quantum number'
        energystring = '::    CMS-PDFT Root'
        stateindex = 3
        enindex = 6

    module = False
    correct_mult = False
    for i, line in enumerate(out):
        if modulestring in line:
            module = True
        elif spinstring in line and module:
            spin = float(line.split()[3])
            if int(2 * spin) + 1 == mult:
                correct_mult = True
        elif energystring in line and module and correct_mult:
            if method == 0 and dkh:
                l = out[i + 4 + state].split()
                return float(l[1])
            else:
                l = line.split()
                if int(l[stateindex]) == state:
                    return float(l[enindex])
    print('CI energy of state %i in mult %i not found!' % (state, mult))
    sys.exit(19)


# ======================================================================= #
def getsmate(out, mult, state1, state2, states):
    # one case:
    # - Dipole moments are in RASSI calculation with two JOBIPH files of same multiplicity

    modulestring = '&RASSI'
    spinstring = 'SPIN MULTIPLICITY:'
    stopstring = 'The following data are common to all the states'
    stop2string = 'MATRIX ELEMENTS OF 1-ELECTRON OPERATORS'
    statesstring = 'Nr of states:'
    matrixstring = 'OVERLAP MATRIX FOR THE ORIGINAL STATES:'

    # first, find the correct RASSI output section for the given multiplicity
    module = False
    jobiphmult = []

    for iline, line in enumerate(out):
        if modulestring in line:
            module = True
            jobiphmult = []
        elif module:
            if spinstring in line:
                jobiphmult.append(int(line.split()[-1]))
            if stopstring in line:
                if jobiphmult == [mult, mult]:
                    break
                else:
                    module = False
    else:
        print('Overlap element not found!', mult, state1, state2)
        print('No correct RASSI run for multiplicity %i found!' % (mult))
        sys.exit(26)

    # Now start searching at iline, looking for the requested matrix element
    for jline, line in enumerate(out[iline + 1:]):
        if stop2string in line:
            print('Overlap element not found!', mult, state1, state2)
            print('Found correct RASSI run, but too few matrix elements!')
            sys.exit(27)
        if statesstring in line:
            nstates = int(line.split()[-1])
        if matrixstring in line:
            rowshift = 1
            for i in range(nstates // 2 + state2 - 1):
                rowshift += i // 5 + 1
            rowshift += 1 + (state1 - 1) // 5
            colshift = (state1 - 1) % 5

            return float(out[iline + jline + rowshift + 1].split()[colshift])


# ======================================================================= #
def get_grad_nac(NACoutfile, nmstates, natom):
    '''
    Creates dictionaries of gradient and nacdr <psi|psi'> matrix.
    Uses ``ParseFile_QCHEM`` class for QCHEM output parsing.
    :param NACoutfile: QCHEM_nac_all
    :param nmstates:
    :param natom:
    :return:
    '''
    _qchem_obj = ParseFile_MOLCAS.ParseFile(NACoutfile, nmstates, natom)

    grad = {}
    nac = {}

    for j in range(natom):
        _OBJX = ParseFile_MOLCAS.ParseFile.gradient_mat(_qchem_obj, j, 'x')
        _OBJY = ParseFile_MOLCAS.ParseFile.gradient_mat(_qchem_obj, j, 'y')
        _OBJZ = ParseFile_MOLCAS.ParseFile.gradient_mat(_qchem_obj, j, 'z')

        grad[j] = {'x': _OBJX[0].diagonal().real,
                   'y': _OBJY[0].diagonal().real,
                   'z': _OBJZ[0].diagonal().real}
        nac[j] = {'x': _OBJX[1],
                  'y': _OBJY[1],
                  'z': _OBJZ[1]}
    return grad, nac


# SBK added this overlap based algorithm.
def _overlap_tracking(QMin, RotMATold, RotMATnew):
    ReigVc_old = RotMATold['Reigvc']
    ReigVc = RotMATnew['Reigvc']

    act_states = []
    maxo_act_states = []
    nroots = QMin['template']['roots'][0]
    for lst in QMin['template']['act_states']:
        maxo = 0.0
        for cst in range(nroots):
            ov = np.abs(np.dot(abs(ReigVc[:, cst].real), abs(ReigVc_old[:, lst - 1])))
            if DEBUG:
                print("Root overlap: %s with %s : %8.6f" % (lst, cst + 1, ov))
            if ov > maxo:
                maxo = ov
                cur = cst
        maxo_act_states.append(maxo)
        act_states.append(cur + 1)
        if DEBUG:
            print("=> Root with max overlap: %s with %s : %8.6f" % (lst, cur + 1, maxo))

    RotMATfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
    if DEBUG:
        print('File created:\t==>\t%s' % (RotMATfile))

    # SBK is now changing the MOLCAS.template act_states keyword and rewriting the file.
    _replaceQMtemplate_line('act_states', act_states)
    if DEBUG:
        print('File modified:\t==>\t%s' % ('MOLCAS.template'))
    # return act_states, maxo_act_states
    return act_states


def _neutral_tracking_WfOverlap(QMin):
    neutral_state_old = int(QMin['template']['neutral_state'])

    with open(os.path.join(QMin['scratchdir'], 'TRACK', "dyson.out")) as f:
        out = f.readlines()

    # out = readfile(os.path.join(QMin['scratchdir'], 'TRACK', "dyson.out"))
    allSTATE_ovlp = wfoverlap_mat(out, "Orthonormalized overlap", QMin)
    allSTATE_ovlp = lowdin(allSTATE_ovlp)

    state_overlap_block_row = abs(allSTATE_ovlp[neutral_state_old - 1, :])
    neutral_state = np.where(state_overlap_block_row == state_overlap_block_row.max())[0][0]
    return neutral_state + 1


def _neutral_tracking(QMin):
    '''
    Params:
        dict : QMin
    Output: neutral_state+1
        Integer : neutral state index.

    Tracks neutral state by using RASSI overlap program in MOLCAS.
    '''

    _nroots = QMin['template']['roots'][0]
    neutral_state_old = int(QMin['template']['neutral_state'])
    h5file = os.path.join(QMin['scratchdir'], 'TRACK', "MOLCAS.rassi.h5")

    with h5py.File(h5file, 'r') as f:
        dataset = f['ORIGINAL_OVERLAPS']
        if 'track_all' in QMin and 'adaptive_track' not in QMin:
            _state_overlap = abs(dataset[:].reshape((_nroots * 2, _nroots * 2)))
            _state_overlap_block = _state_overlap[_nroots:, :_nroots]
            state_overlap_block_row = _state_overlap_block[neutral_state_old - 1, :]
            neutral_state = np.where(state_overlap_block_row == state_overlap_block_row.max())[0][0]
            return neutral_state + 1
        else:
            state_overlap = abs(dataset[:].reshape((_nroots + 1, _nroots + 1)))
            neutral_state = np.where(state_overlap[:_nroots, _nroots] == state_overlap[:_nroots, _nroots].max())[0][0]
            return neutral_state + 1


def lowdin(S):
    _U, _D, _VT = np.linalg.svd(S, full_matrices=True)
    S_ortho = _U @ _VT
    return S_ortho


def _tracking_with_natorb(QMin, RotMATnew, natorbs_new):
    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
    nroots = QMin['template']['roots'][0]

    kabsch = False
    while True:
        # Need mixaoovl
        _DMOLWORKDIR = os.path.join(QMin['scratchdir'], 'DOUBLEMOL')
        errorcode, _ = run_double_mol(_DMOLWORKDIR, QMin, kabsch, guessorb=True)
        mixaoovlfile = os.path.join(_DMOLWORKDIR, "MOLCAS.guessorb.h5")
        with h5py.File(mixaoovlfile, 'r') as file:
            dataset = file['AO_OVERLAP_MATRIX'][:]
            dimension = int(np.sqrt(dataset.shape[0] / 4))
            mixaoovl = dataset.reshape(2 * dimension, 2 * dimension)[:dimension, dimension:]
        if np.allclose(mixaoovl, np.zeros(mixaoovl.shape), 1E-6):
            kabsch = True
        if kabsch == False:
            break

    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]
        natorb_old = f["natorb"][:]

    nelec = QMin['template']['inactive'] * 2 + QMin['template']['nactel']

    for lst in QMin['template']['act_states']:
        _ovlp = np.array([abs(reduce(np.dot, (
        np.asarray(natorb_old)[lst][:, int(nelec / 2)], mixaoovl, np.asarray(natorbs_new)[istate][:, int(nelec / 2)])))
                          for istate in range(nroots)])
        cur = np.where(_ovlp == _ovlp.max())[0][0] + 1
        act_states.append(cur + 1)

    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
        f.create_dataset("natorb", data=natorbs_new)

    if DEBUG:
        print('File created:\t==>\t%s' % (RotMATfile))

    _replaceQMtemplate_line('act_states', act_states)
    if DEBUG:
        print('File modified:\t==>\t%s' % ('MOLCAS.template'))
    return act_states


def _tracking_with_RASSI(QMin, RotMATnew, lowdinSVD=False):
    """
    Tracks electronic states using RASSI overlaps from OpenMOLCAS.
    This method does not account for mixed atomic orbital (AO) overlaps.

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
        Index of the electronic state with the highest overlap.
    maxo : float
        Maximum overlap value corresponding to the tracked state.

    :rtype: tuple (int, float)
    """

    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
    nroots = QMin['template']['roots'][0]
    h5file = os.path.join(QMin['scratchdir'], 'TRACK', "MOLCAS.rassi.h5")

    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]

    with h5py.File(h5file, 'r') as file:
        if 'ORIGINAL_OVERLAPS' in file:
            dataset = file['ORIGINAL_OVERLAPS']
            allSTATE_ovlp = dataset[:].reshape((nroots * 2, nroots * 2))[nroots:, :nroots]

    _ReigVc_new = RotMATnew['Reigvc']
    _LeigVc_old = RotMATold['Leigvc']
    _LeigVc_new = RotMATnew['Leigvc']
    _ReigVc_old = RotMATold['Reigvc']

    if lowdinSVD:
        allSTATE_ovlp = lowdin(allSTATE_ovlp)
        if 'xms_correction' in QMin:
            allSTATE_ovlp = reduce(np.dot, (QMin['u_xms'].T, allSTATE_ovlp, QMin['u_xms']))

    allSTATE_ovlp_DIAG = _LeigVc_old.T @ allSTATE_ovlp @ _ReigVc_new

    if lowdinSVD:
        allSTATE_ovlp_DIAG = lowdin(allSTATE_ovlp_DIAG)

    maxo = []
    for lst in (QMin['template']['act_states']):
        cur_max = np.where(abs(allSTATE_ovlp_DIAG[lst - 1, :]) == abs(allSTATE_ovlp_DIAG[lst - 1, :]).max())[0][0]
        act_states.append(cur_max + 1)
        maxo.append(allSTATE_ovlp_DIAG[lst - 1, :][cur_max])

    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
    if DEBUG:
        print('File created:\t==>\t%s' % (RotMATfile))

    _replaceQMtemplate_line('act_states', act_states)
    if DEBUG:
        print('File modified:\t==>\t%s' % ('MOLCAS.template'))
    return act_states, maxo


def _tracking_with_WFOVERLAP(QMin, RotMATnew, lowdinSVD=False):
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

    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]

    with open(os.path.join(QMin['scratchdir'], 'TRACK', "dyson.out")) as f:
        out = f.readlines()

    allSTATE_ovlp = wfoverlap_mat(out, "Orthonormalized overlap", QMin)
    if lowdinSVD:
        allSTATE_ovlp = lowdin(allSTATE_ovlp)
        if 'xms_correction' in QMin:
            allSTATE_ovlp = reduce(np.dot, (QMin['u_xms'].T, allSTATE_ovlp, QMin['u_xms']))

    _LeigVc_old = RotMATold['Leigvc']
    _ReigVc_new = RotMATnew['Reigvc']
    _ReigVc_old = RotMATold['Reigvc']
    _LeigVc_new = RotMATnew['Leigvc']

    allSTATE_ovlp_DIAG = reduce(np.dot, (_LeigVc_old.T, allSTATE_ovlp, _ReigVc_new))  # SBK (02/29/2024)
    # July 19, 2024

    if lowdinSVD:
        allSTATE_ovlp_DIAG = lowdin(allSTATE_ovlp_DIAG)

    maxo = []
    for lst in (QMin['template']['act_states']):
        cur_max = np.where(abs(allSTATE_ovlp_DIAG[lst - 1, :]) == abs(allSTATE_ovlp_DIAG[lst - 1, :]).max())[0][0]
        act_states.append(cur_max + 1)
        maxo.append(allSTATE_ovlp_DIAG[lst - 1, :][cur_max])

    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
    if DEBUG:
        print('File created:\t==>\t%s' % (RotMATfile))

    _replaceQMtemplate_line('act_states', act_states)
    if DEBUG:
        print('File modified:\t==>\t%s' % ('MOLCAS.template'))
    return act_states, maxo


def align_geometry_Kabsch(oldgeom, newgeom, QMin):
    """
    Aligns two sets of points using the Kabsch algorithm for rigid-body transformation (translation and rotation).
    `newgeom` is aligned to `oldgeom`.

    Parameters:
    ----------
    oldgeom : numpy.ndarray, shape (N, 3)
        Reference geometry (fixed coordinates).
    newgeom : numpy.ndarray, shape (N, 3)
        Geometry to be aligned.

    Returns:
    ----------
    aligned_geom : numpy.ndarray, shape (N, 3)
        Aligned version of `newgeom` after applying translation and rotation.
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

    atom_symbols = [i[0] for i in newgeom]

    # Align geom2 to geom1
    aligned_newgeom = align_geometries([i[1:] for i in oldgeom], [i[1:] for i in newgeom])  # Strip them off the labels

    coords_aligned = []
    for idx, coord in enumerate(aligned_newgeom):
        x, y, z = coord
        coords_aligned.append((atom_symbols[idx], x, y, z))

    # Print the aligned geometry
    print("===> Kabsch algorithm is invoked\nNew geometry (before alignment)!\n")
    string = 'Geometry in Angstrom:\n'
    for i in range(QMin['natom']):
        string += '%s ' % (QMin['geo'][i][0])
        for j in range(3):
            string += '% 7.4f ' % (newgeom[i][j + 1])
        string += '\n'
    print(string)

    print("===> New geometry (after alignment)!\n")
    string = 'Geometry in Angstrom:\n'
    for i in range(QMin['natom']):
        string += '%s ' % (QMin['geo'][i][0])
        for j in range(3):
            string += '% 7.4f ' % (coords_aligned[i][j + 1])
        string += '\n'
    print(string)

    return coords_aligned


def need_alignment(WORKDIR, QMin):
    """
    Confirm whether alignment is needed

    Parameters:
    ----------
    WORKDIR: str
          workdir of wfoverlap directory

    QMin: dict
        dictionary containing QM run information

    Returns:
    -------
    rerunWFOVLP: bool
            True Need alignment, False No alignment needed
    """

    nroots = QMin['template']['roots'][0]
    inputfile = os.path.join(WORKDIR, 'dyson.in')
    outfile = os.path.join(WORKDIR, 'dyson.out')
    out = readfile(outfile)
    # string = readfile(inputfile)
    rerunWFOVLP = False

    allSTATE_ovlp = wfoverlap_mat(out, "Overlap", QMin)
    # If overlap matrix is all zeros
    if np.allclose(allSTATE_ovlp, np.zeros([nroots, nroots]), 1E-8):
        rerunWFOVLP = True
        '''
        print("\n\n===> Warning: WFOVERLAP run yielded Overlap matrix with all zero elements\n"
              "Possible bug with mix_aoovl, performing same_aos calculation subsequently!\n\n")
        '''
        print("\n\n===> Warning: WFOVERLAP run yielded Overlap matrix with all zero elements\n"
              "Consecutive geometries might be translationally and/or rotationally away from each other!\n")

        # string = [line.replace('ao_read=1', 'ao_read=-1') for line in string]
        # string.append("same_aos\n")

    return rerunWFOVLP


def setupWfOverlapDIR_direct(WORKDIR, QMin, RotMATold, RotMATnew):
    """
    1. Sets up a calculation in an already existing directory.
    Rotates CI vectors by PCAP rotation matrices and feeds it to WF-OVERLAP
    program and directly calculates WF-OVLP in DIAG basis.

    2. Directly generates both the real and imaginary CI-vectors and
    add both contributions to total approximated overlap (real)

    This may not be needed in future

    Parameters:
    ----------
    WORKDIR: str
        Working directory of WF-OVERLAP
    QMin: dict
            dictionary containing QM run information
    RotMATold: dict
            dictionary containing old rotation matrix
    RotMATnew: dict
            dictionary containing new rotation matrix


    Returns:
    -------
    A directory for WF-OVERLAP calculation of individual terms (real and imaginary, real-imaginary)
    """
    wfovout = os.path.join(QMin['scratchdir'], 'TRACK', "dyson.out")
    nroots = QMin['template']['roots'][0]
    detsfileold = os.path.join(WORKDIR, "dets.old")
    detsfilenew = os.path.join(WORKDIR, "dets.new")
    dets_diag_old_real = os.path.join(WORKDIR, "dets.diag.old.real")
    dets_diag_old_imag = os.path.join(WORKDIR, "dets.diag.old.imag")
    dets_diag_new_real = os.path.join(WORKDIR, "dets.diag.new.real")
    dets_diag_new_imag = os.path.join(WORKDIR, "dets.diag.new.imag")

    LeigvcO = RotMATold['Leigvc']
    ReigvcO = RotMATold['Reigvc']
    LeigvcN = RotMATnew['Leigvc']
    ReigvcN = RotMATnew['Reigvc']

    print_civecs_diag(nroots, LeigvcO, ReigvcO, detsfileold, dets_diag_old_real)
    print_civecs_diag(nroots, LeigvcO, ReigvcO, detsfileold, dets_diag_old_imag, dtype='imag')

    print_civecs_diag(nroots, LeigvcN, ReigvcN, detsfilenew, dets_diag_new_real)
    print_civecs_diag(nroots, LeigvcN, ReigvcN, detsfilenew, dets_diag_new_imag, dtype='imag')

    inputfile = os.path.join(WORKDIR, 'dyson.in')
    lstring_orig = readfile(inputfile)
    lstring = copy.deepcopy(lstring_orig)

    # Real part
    lstring = [line.replace('a_det=dets.old', 'a_det=dets.diag.old.real') for line in lstring]
    lstring = [line.replace('b_det=dets.new', 'b_det=dets.diag.new.real') for line in lstring]
    writefile(inputfile, lstring)
    runerror = runWFOVERLAPS(WORKDIR, QMin['wfoverlap'], QMin['memory'], QMin['ncpu_avail'])

    # COPY
    if runerror == 0:
        shutil.copy(wfovout, os.path.join(QMin['scratchdir'], 'TRACK', "wfov.out.real"))

    # Imaginary part
    lstring = copy.deepcopy(lstring_orig)
    lstring = [line.replace('a_det=dets.old', 'a_det=dets.diag.old.imag') for line in lstring]
    lstring = [line.replace('b_det=dets.new', 'b_det=dets.diag.new.imag') for line in lstring]
    writefile(inputfile, lstring)
    runerror = runWFOVERLAPS(WORKDIR, QMin['wfoverlap'], QMin['memory'], QMin['ncpu_avail'])
    if runerror == 0:
        shutil.copy(wfovout, os.path.join(WORKDIR, "wfov.out.imag"))

    # Real-Imaginary part
    lstring = copy.deepcopy(lstring_orig)
    lstring = [line.replace('a_det=dets.old', 'a_det=dets.diag.old.real') for line in lstring]
    lstring = [line.replace('b_det=dets.new', 'b_det=dets.diag.new.imag') for line in lstring]
    writefile(inputfile, lstring)
    runerror = runWFOVERLAPS(WORKDIR, QMin['wfoverlap'], QMin['memory'], QMin['ncpu_avail'])
    if runerror == 0:
        shutil.copy(wfovout, os.path.join(WORKDIR, "wfov.out.reim"))

    # Imaginary-Real part
    lstring = copy.deepcopy(lstring_orig)
    lstring = [line.replace('a_det=dets.old', 'a_det=dets.diag.old.imag') for line in lstring]
    lstring = [line.replace('b_det=dets.new', 'b_det=dets.diag.new.real') for line in lstring]
    writefile(inputfile, lstring)
    runerror = runWFOVERLAPS(WORKDIR, QMin['wfoverlap'], QMin['memory'], QMin['ncpu_avail'])
    if runerror == 0:
        shutil.copy(wfovout, os.path.join(WORKDIR, "wfov.out.imre"))


def _tracking_with_WFOVERLAP_direct(QMin, RotMATnew, lowdinSVD=False):
    """
    Tracks electronic states using RASSI overlaps from WF-OVERLAP.
    Uses setupWfOverlapDIR_direct program (see that for more details)

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
        Index of the electronic state with the highest overlap.
    maxo : float
        Maximum overlap value corresponding to the tracked state.

    :rtype: tuple (int, float)
    """
    act_states = []
    RotMATold = {}

    RotMATfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
    with h5py.File(RotMATfile, "r") as f:
        RotMATold['Reigvc'] = f["Reigvc"][:]
        RotMATold['Leigvc'] = f["Leigvc"][:]

    # Set up and PERFORM direct wf-ovlp calculation
    WORKDIR = os.path.join(QMin['scratchdir'], 'TRACK')
    setupWfOverlapDIR_direct(WORKDIR, QMin, RotMATold, RotMATnew)

    # Read real part
    with open(os.path.join(QMin['scratchdir'], 'TRACK', "wfov.out.real")) as f:
        out_real = f.readlines()

    # Read imaginary part
    with open(os.path.join(QMin['scratchdir'], 'TRACK', "wfov.out.imag")) as f:
        out_imag = f.readlines()

    # Read real-imaginary part
    with open(os.path.join(QMin['scratchdir'], 'TRACK', "wfov.out.reim")) as f:
        out_reim = f.readlines()

    # Read imaginary-real part
    with open(os.path.join(QMin['scratchdir'], 'TRACK', "wfov.out.imre")) as f:
        out_imre = f.readlines()

    allSTATE_ovlp_imag = wfoverlap_mat(out_imag, "Overlap", QMin)
    allSTATE_ovlp_real = wfoverlap_mat(out_real, "Overlap", QMin)
    allSTATE_ovlp_reim = wfoverlap_mat(out_reim, "Overlap", QMin)
    allSTATE_ovlp_imre = wfoverlap_mat(out_imre, "Overlap", QMin)

    # Add them together
    allSTATE_ovlp_DIAG = allSTATE_ovlp_real - allSTATE_ovlp_imag + 1.0j * (allSTATE_ovlp_reim + allSTATE_ovlp_imre)

    if lowdinSVD:
        allSTATE_ovlp_DIAG = lowdin(allSTATE_ovlp_DIAG)

    maxo = []
    for lst in (QMin['template']['act_states']):
        cur_max = np.where(abs(allSTATE_ovlp_DIAG[lst - 1, :]) == abs(allSTATE_ovlp_DIAG[lst - 1, :]).max())[0][0]
        act_states.append(cur_max + 1)
        maxo.append(allSTATE_ovlp_DIAG[lst - 1, :][cur_max])

    # Replace with new rotation matrix
    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
        f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
    if DEBUG:
        print('File created:\t==>\t%s' % (RotMATfile))

    _replaceQMtemplate_line('act_states', act_states)
    if DEBUG:
        print('File modified:\t==>\t%s' % ('MOLCAS.template'))

    return act_states, maxo


def _extract_overlap_mat(h5file, nroots):
    """
    Extracts overlap matrix from MOLCAS.rassi.h5

    Parameters:
    ----------
    h5file
        MOLCAS.rassi.h5 filename
    nroots
        Dimension of the overlap matrix row
    Returns:
    -------
    Overlap matrix (nroots x nroots)

    """
    with h5py.File(h5file, 'r') as file:
        if 'ORIGINAL_OVERLAPS' in file:
            dataset = file['ORIGINAL_OVERLAPS']
            try:
                allSTATE_ovlp = dataset[:].reshape((nroots * 2, nroots * 2))[nroots:, :nroots]
            except:
                allSTATE_ovlp = dataset[:].reshape((nroots + 1, nroots + 1))
    return allSTATE_ovlp


def _track_w_adaptive_reference(QMin, RotMATnew, lowdinSVD=False):
    """
    Adaptive tracking from the following paper
    https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26390
    Updates the reference state if needed

    Parameters:
    ----------
    QMin
    RotMATnew
    lowdinSVD

    Returns:
    -------
    act_states : int
        Index of the electronic state with the highest overlap.
    update_ref : bool
        If we should update the reference or not.

    :rtype: tuple (int, bool)
    """
    _update_ref = []
    RotMATold = {}
    act_states = []
    _max_thrs, _lower_ratio_thrs, _upper_ratio_thrs = [0.5, 0.3, 0.6]
    nroots = QMin['template']['roots'][0]

    _ReigVc_new = RotMATnew['Reigvc']
    for _ct, lst in enumerate(QMin['template']['act_states']):
        h5file = os.path.join(QMin['scratchdir'], 'TRACK_%s' % lst, "MOLCAS.rassi.h5")
        allSTATE_ovlp = _extract_overlap_mat(h5file, nroots)
        rotmatfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5.%s" % (_ct + 1))

        with h5py.File(rotmatfile, "r") as f:
            RotMATold['Reigvc'] = f["Reigvc"][:]
            RotMATold['Leigvc'] = f["Leigvc"][:]

        _LeigVc_old = RotMATold['Leigvc']
        allSTATE_ovlp_DIAG = _LeigVc_old.T.real @ allSTATE_ovlp @ _ReigVc_new.real
        if lowdinSVD:
            allSTATE_ovlp_DIAG = lowdin(allSTATE_ovlp_DIAG)

        _update_ref.append(False)
        _ovlp = abs(allSTATE_ovlp_DIAG)[lst - 1, :]
        _idx = _ovlp.argsort()[::-1]
        act_states.append(_idx[0] + 1)

        maxo = _ovlp[_idx[0]]
        maxo_2nd = _ovlp[_idx[1]]

        print("\n\n\nMaximum state overlap with old state: %i is %10.6f\n" % (lst, maxo))
        print("2nd Maximum state overlap is %10.6f\n" % (maxo_2nd))
        print("The ratio of state: %i and state: %i overlap values : %6.3f\n\n\n" % (
            _idx[0] + 2, _idx[0] + 1, (maxo_2nd / maxo)))

        if maxo > _max_thrs:
            if maxo_2nd / maxo > _lower_ratio_thrs and maxo_2nd / maxo < _upper_ratio_thrs:
                _update_ref[-1] = True

        # Now check if we need to update the reference (SCF or fixed one)
        with h5py.File(rotmatfile, "w") as f:
            if not _update_ref[_ct]:
                f.create_dataset("Reigvc", data=RotMATold['Reigvc'])
                f.create_dataset("Leigvc", data=RotMATold['Leigvc'])
            else:
                # May not be required, already done before in _tracking_rassi module, a default choice (SCF)
                f.create_dataset("Reigvc", data=RotMATnew['Reigvc'])
                f.create_dataset("Leigvc", data=RotMATnew['Leigvc'])
    _replaceQMtemplate_line('act_states', act_states)
    _replaceQMtemplate_line('update_ref', _update_ref)
    return act_states, _update_ref


# SBK: Replace some lines in MOLCAS.template
def _replaceQMtemplate_line(string, inpArr):
    _templateName = 'MOLCAS.template'
    template = readfile(_templateName)
    for i, line in enumerate(template):
        if line.startswith(string):
            new_line = f"{string} {' '.join(map(str, inpArr))}\n"
            template[i] = new_line
    with open(_templateName, "w") as f:
        f.writelines(template)


def _addQMtemplate_line(string, inpArr):
    _templateName = 'MOLCAS.template'
    new_line = f"{string} {' '.join(map(str, inpArr))}\n"
    with open(_templateName, "a") as f:
        f.writelines(new_line)


def wfoverlap_mat(out, matrix_type, QMin):
    nroots = QMin['template']['roots'][0]
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


_dm_AO_total_molcas = lambda k, l, pcOBJ: pcOBJ.get_density(k, l, False) + pcOBJ.get_density(k, l, True)


def _generate_1RDM_diag_molcas(pcOBJ, Rvec, LVec):
    """
    Genartes 1-RDM in DIAG basis

    Parameters:
    ----------
    pcOBJ: Projected-CAP object (generated via opencap)

    Rvec: Rotation vector (right)

    LVec: Rotation vector (left)

    Returns:
    -------
    1-RDM in DIAG basis
    """
    nroots = np.shape(LVec)[1]
    _1RDM_diag_molcas = {}
    for res_state1 in range(nroots):
        for res_state2 in range(nroots):
            _density = 0.0
            for k in range(nroots):
                for l in range(nroots):
                    _density += LVec.T[res_state1, k] * _dm_AO_total_molcas(k, l, pcOBJ) * Rvec[l, res_state2]
            _1RDM_diag_molcas[res_state1, res_state2] = _density.copy()
    return _1RDM_diag_molcas


def orthoMO(C, S):
    '''
    Adapted from MOLCAS, normalizing C to C', so that C'^TSC' is Identity matrix.
    '''
    W = reduce(np.dot, (C.T, S, C))
    W_sqrt = LA.sqrtm(W)
    W_inv_sqrt = LA.inv(W_sqrt)
    Cprime = reduce(np.dot, (C, W_inv_sqrt))
    return Cprime


def _generate_NaturalOrbs(S, nroots, _1RDM_diag_molcas):
    """
    Genartes natural orbitals from the DIAG 1-RDMs

    Parameters:
    ----------
    S: AO overlap matrix

    nroots: No of roots

    _1RDM_diag_molcas: 1-RDM (in DIAG basis) that needs to diagonalized

    Returns:
    -------
    noons_all: natural orbitals' occupations

    natorbs_all: natural orbitals (only real part)
    """
    noons_all = []
    natorbs_all = []
    for i in range(nroots):
        A = reduce(np.dot, (_1RDM_diag_molcas[i, i], S))
        w, v = LA.eig(A)
        v = orthoMO(v, S)
        _des_idx = np.argsort(w.real)[::-1]
        noons = w[_des_idx].real
        natorbs = v[:, _des_idx].real
        noons_all.append(noons)
        natorbs_all.append(natorbs)
    return noons_all, natorbs_all


def _generate_molden_file_w_pySCF(flines, fMOLDEN, natorbs, noons, QMin):
    """
    Genrates molden file (for resonance state)
    (uses pyscf.tools.molden to print orbitals)

    Parameters:
    ----------
    flines: lines continuing geometry info

    fMOLDEN: MOLDEN file name

    natorbs: Natural orbitals

    noons: occupation numbers

    QMin: QMin dictionaries

    Returns:
    -------
    prints MOLDEN file ('natorb' keyword should be invoked in the MOLCAS.template file)

    """

    def _xyz4pyscf(lines):
        atom_count = int(QMin['natom']) + 1 if 'ghost' in QMin else int(QMin['natom'])
        coords = []
        for _idx, line in enumerate(lines):
            if "Cartesian coordinates" in line:
                for _xyzline in (lines[_idx + 4:_idx + 4 + atom_count]):
                    columns = _xyzline.split()
                    label, x, y, z = columns[1:]
                    coords.append((label, float(x), float(y), float(z)))
                break
        return coords

    def atom_labelfree(xyzdict):
        unique_labels = []
        for label, _, _, _ in xyzdict:
            unique_label = ''.join(filter(str.isalpha, label))
            if unique_label not in unique_labels:
                unique_labels.append(unique_label)
        return unique_labels

    from pyscf.gto.basis import parse_gaussian
    from pyscf import gto
    from pyscf.tools.molden import header, orbital_coeff

    atomdict = _xyz4pyscf(flines)
    atomdict_idxfree = atom_labelfree(atomdict)
    # [atom[0][:-1] for _, atom in enumerate(atomdict)]
    gbspath = '%s/OPENCAPMD.gbs' % str(QMin['pwd'])
    gbas_dict = {}
    for _, atomlabel in enumerate(atomdict_idxfree):
        gbas_dict[atomlabel[0]] = parse_gaussian.load(gbspath, atomlabel[0])

    mol_info = gto.M(atom=atomdict, basis=gbas_dict, symmetry=1, verbose=0)

    '''
    _1DM_diag = _generate_DM_diag_molcas(pcOBJ, Reigvec, Leigvec)
    noons_all, natorbs_all = _generate_NaturalOrbs(AOoverlap, nroots, _1DM_diag)
    '''
    with open(fMOLDEN, 'w') as f1:
        header(mol_info, f1)
        cgto_molcas_rank = np.arange(0, mol_info.nao, 1)
        try:
            orbital_coeff(mol_info, f1, natorbs, ene=np.arange(0, mol_info.nao, 1, dtype=float), occ=noons,
                          aoidx=cgto_molcas_rank)
            '''with some in house modifications of molden.py file in pyscf/tools'''
        except:
            orbital_coeff(mol_info, f1, natorbs, ene=np.arange(0, mol_info.nao, 1, dtype=float), occ=noons)


def read_calc_neutral(out, gradnacfile=True):
    """
    Parses neutral energy from outputfile
    :param out: list of string
    :return: float, energy value in Hartree
    """
    neutral = None
    if gradnacfile:
        for idx, line in enumerate(out):
            if "Merging NEUTRAL" in line:
                for idxf, linef in enumerate(out[idx:]):
                    if 'RASSCF root' in linef:
                        neutral = float(linef.split()[-1])
                        break
    else:
        for idxf, linef in enumerate(out):
            if 'RASSCF root' in linef:
                neutral = float(linef.split()[-1])
                break
    return neutral


def print_to_table(title, headers, data, align=None):
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

    print(f"\n{title.center(10)}\n")
    #    table = tabulate(string, headers=headers, tablefmt='pretty', numalign="center", stralign="center")
    table = tabulate(string, headers=headers, tablefmt='pretty', numalign="center", stralign="center", floatfmt='g')
    if align:
        table = tabulate(string, headers=headers, tablefmt='pretty', numalign="center", stralign=align, floatfmt='g')

    print(table)


def getQMout(out, QMin, outfilename=None):  # SBK added new outfilename args, default is None
    '''Constructs the requested matrices and vectors using the get<quantity> routines.

    The dictionary QMout contains all the requested properties. Its content is dependent on the keywords in QMin:
    - 'h' in QMin:
                    QMout['h']: list(nmstates) of list(nmstates) of complex, the non-relaticistic hamiltonian
    - 'soc' in QMin:
                    QMout['h']: list(nmstates) of list(nmstates) of complex, the spin-orbit hamiltonian
    - 'dm' in QMin:
                    QMout['dm']: list(3) of list(nmstates) of list(nmstates) of complex, the three dipole moment matrices
    - 'grad' in QMin:
                    QMout['grad']: list(nmstates) of list(natom) of list(3) of float, the gradient vectors of every state (even if "grad all" was not requested, all nmstates gradients are contained here)
    - 'nac' in QMin and QMin['nac']==['num']:
                    QMout['nac']: list(nmstates) of list(nmstates) of complex, the non-adiabatic coupling matrix
                    QMout['mrcioverlap']: list(nmstates) of list(nmstates) of complex, the MRCI overlap matrix
                    QMout['h']: like with QMin['h']
    - 'nac' in QMin and QMin['nac']==['ana']:
                    QMout['nac']: list(nmstates) of list(nmstates) of list(natom) of list(3) of float, the matrix of coupling vectors
    - 'nac' in QMin and QMin['nac']==['smat']:
                    QMout['nac']: list(nmstates) of list(nmstates) of complex, the adiabatic-diabatic transformation matrix
                    QMout['mrcioverlap']: list(nmstates) of list(nmstates) of complex, the MRCI overlap matrix
                    QMout['h']: like with QMin['h']

    Arguments:
    1 list of strings: Concatenated MOLCAS output
    2 dictionary: QMin
    3 filename with extensions: default is None needed for Projected-CAP

    Returns:
    1 dictionary: QMout'''

    # get version of MOLCAS
    version = QMin['version']
    method = QMin['method']

    # Currently implemented keywords: h, soc, dm, grad, nac (num,ana,smat)
    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    nroots = QMin['template']['roots'][0]
    QMout = {}

    # h: get CI energies of all ci calculations and construct hamiltonian, returns a matrix(nmstates,nmstates)
    if 'screening' not in QMin['template']:
        H0, H_CAP, pcapOBJ, systemOBJ = parse_MOLCAS_Hamiltonian(outfilename, QMin, do_numerical=True, saveRASSI=True)
    else:
        _h5file = os.path.join(QMin['scratchdir'], 'TRD1', 'SAVE.HCAP.h5')
        with h5py.File(_h5file, "r") as f:
            H0 = f["H0"][:]
            H_CAP = f["H_CAP"][:]

    QMout['h'] = H0

    if 'xms_correction' in QMin:
        QMin['u_xms'] = readXMStransform(QMin, os.path.join(QMin['scratchdir'], 'xms_corr', "MOLCAS.out"))
    else:
        QMin['u_xms'] = np.eye(len(H0))

    if 'pcap' in QMin:
        QMout['h'] = H0 + float(QMin['template']['cap_eta']) * (1E-5) * H_CAP

    # Create Left and Right eigen vectors for the non-hermitian Hamiltonian (H).
    # Get H in diag basis
    # Leigvc, Reigvc, eVs = LUdiagonalize(QMout['h'])
    Leigvc, Reigvc, eVs = diagonalize_cmatrix(QMout['h'])

    RotMATnew = {}
    RotMATnew['Reigvc'] = Reigvc
    RotMATnew['Leigvc'] = Leigvc

    # RotMATold = {}
    # get states that we need to run SH
    # Now not using it : SBK
    # act_states, R2AVG = find_lowest_r_squared(outfilename, Leigvc, Reigvc, nroots)
    # the rotation matrix file for `ZO' to `diag' transformation is looked for in the following file.
    # rotmatfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
    maxo = None
    try:
        '''
        with h5py.File(rotmatfile, "r") as f:
            RotMATold['Reigvc'] = f["Reigvc"][:]
            RotMATold['Leigvc'] = f["Leigvc"][:]
        '''
        if 'track_all' in QMin:
            if 'adaptive_track' in QMin:
                print("\n\n==>\tThis is RASSI overlap based tracking for resonance state(s) with adaptive tracking ON!")
                act_states, _update_ref = _track_w_adaptive_reference(QMin, RotMATnew, lowdinSVD=False)
            else:
                act_states, maxo = _tracking_with_RASSI(QMin, RotMATnew,
                                                        lowdinSVD=False)  # I am setting it to false. SBK 02/29/24
                print("\n\n==>\tThis is RASSI overlap based tracking for resonance state(s)!")
        elif 'track_wfoverlap' in QMin:

            act_states, maxo = _tracking_with_WFOVERLAP(QMin, RotMATnew,
                                                        lowdinSVD=False)  # I am setting it to false. SBK 02/29/24

            print("\n\n==>\tThis is WFOVERLAP overlap based tracking for resonance state(s)!")
        elif 'track_natorb' in QMin:
            _1RDM_diag = _generate_1RDM_diag_molcas(pcapOBJ, Reigvc, Leigvc)
            noon_lib, natorbs_lib = _generate_NaturalOrbs(systemOBJ.get_overlap_mat(), nroots, _1RDM_diag)
            rotmatfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
            with h5py.File(rotmatfile, "w") as f:
                f.create_dataset("natorb", data=natorbs_lib)
            act_states = _tracking_with_natorb(QMin, RotMATnew, natorbs_lib)
            print("\n\n==>\tThis is natural orbital based overlap based tracking for resonance state(s)!")
        else:
            act_states = _overlap_tracking(QMin, RotMATnew)  # Fixed SBK 02/29/24
            print("\n\n==>\tThis is Projected-CAP rotation matrix overlap based tracking for resonance state(s)!")
    except Exception as e:
        print(f"\n ====> An error occurred (check QM.err carefully!): \n\t{e}")
        traceback.print_exc()

        act_states = QMin['template']['act_states']
        if 'adaptive_track' in QMin:
            for _ct, _state in enumerate(act_states):
                rotmatfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5.%s" % (_ct + 1))
                with h5py.File(rotmatfile, "w") as f:
                    f.create_dataset("Reigvc", data=Reigvc)
                    f.create_dataset("Leigvc", data=Leigvc)
        else:
            rotmatfile = os.path.join(QMin['savedir'], "PCAP.RotMatOld.h5")
            with h5py.File(rotmatfile, "w") as f:
                f.create_dataset("Reigvc", data=Reigvc)
                f.create_dataset("Leigvc", data=Leigvc)
        print("\n===>\tWarning: Active state(s) chosen from MOLCAS.template and not by overlap based tracking!\n")
        maxo = np.ones(len(QMin['template']['act_states']), dtype=complex)

    if 'init' in QMin and 'adaptive_track' in QMin:
        act_states = QMin['template']['act_states']
        _update_ref = []
        for _ct, _state in enumerate(act_states):
            _update_ref.append(True)
        _addQMtemplate_line('update_ref', _update_ref)

    if 'natorb' in QMin:
        read_geom_from = out
        if 'grad' not in QMin:
            master_filename = os.path.join(QMin['scratchdir'], 'master', 'MOLCAS.out')
            outgeom = readfile(master_filename)
            read_geom_from = outgeom
            # TODO, fix this mess
        if 'track_natorb' not in QMin:
            _1RDM_diag = _generate_1RDM_diag_molcas(pcapOBJ, Reigvc, Leigvc)
            noon_lib, natorbs_lib = _generate_NaturalOrbs(systemOBJ.get_overlap_mat(), nroots, _1RDM_diag)
        for _actstate in (act_states):
            fMOLDEN = os.path.join(QMin['savedir'], "MOLCAS.DIAG_step%s.NatOrb.%s" % (str(QMin['step'][0]), _actstate))
            try:
                _generate_molden_file_w_pySCF(read_geom_from, fMOLDEN, natorbs_lib[_actstate - 1],
                                              noon_lib[_actstate - 1], QMin)
            except Exception as e:
                print(
                    f"\n ===> An error occurred (Check if 1. pySCF is installed, 2. Basis file for pySCF is in {str(QMin['pwd'])}): \n\t{e}")
                traceback.print_exc()

        # TODO
        '''
        Remove this later: Note to SBK, 
        '''
        for istate in range(nroots):
            #            fMOLDEN = os.path.join(QMin['scratchdir'], 'master', "MOLCAS.DIAG_step%s.NatOrb.%s" % (str(QMin['step'][0]), istate+1))
            _fMOLDENpath = os.path.join(QMin['savedir'], 'MOLDEN')
            os.makedirs(_fMOLDENpath, exist_ok=True)
            fMOLDEN = os.path.join(_fMOLDENpath, "MOLCAS.DIAG_step%s.NatOrb.%s" % (str(QMin['step'][0]), istate + 1))
            try:
                _generate_molden_file_w_pySCF(read_geom_from, fMOLDEN, natorbs_lib[istate], noon_lib[istate], QMin)
            except Exception as e:
                traceback.print_exc()

    del systemOBJ
    del pcapOBJ

    if 'pcap' in QMin:
        title = "Active states (from SA roots)"
        try:
            headers = ["Root (old)", "Root (new)", "Overlap (Re, Im)"]
            data = list(zip(QMin['template']['act_states'], act_states, maxo))  # Convert to a list to reuse

            print_to_table(title, headers, data)
            '''
            headers = ["Root (new)", "Root (old)"]
            data = [(state, state_old) for state, state_old in zip(act_states, QMin['template']['act_states'])]
            table = tabulate(data, headers=headers, tablefmt="fancy_grid", colalign=("center", "center"))
            print(f"\n{title.center(len(table.splitlines()[0]))}")
            print(table)
            '''
        except:
            string = "\n" + "=" * len(title) + f" {title} " + "=" * len(title) + "\n"
            for state, state_old in zip(act_states, QMin['template']['act_states']):
                string += f"Root (new) : {state} \t\t\tRoot (old) : {state_old}\n"
            string += "=" * (2 * len(title) + len(title) + 2)
            print(string)

    # SBK added this
    try:
        title = "Projected CAP generated complex energies (Hartree)"
        headers = ["Root", "Complex energies (Re, Im)"]
        energy_diag = list(
            zip(act_states, [eVs.diagonal()[res_idx - 1] for res_idx in act_states]))  # Convert to a list to reuse
        print_to_table(title, headers, energy_diag)
    except:
        print("\n\n===> Projected CAP generated energies. Needed for SL-FSSH")
        print("+=======+======================+======================+")
        print("| %-5s | %-20s | %-20s |" % ("State", "Re(E, Hartree)", "Im(E, Hartree)"))
        print("+=======+======================+======================+")
        for _, res_idx in enumerate(act_states):
            _re_value = np.real(eVs.diagonal()[res_idx - 1])
            _im_value = np.imag(eVs.diagonal()[res_idx - 1])
            print("| %-5s | %-20.12E | %-20.12E |" % (str(res_idx), _re_value, _im_value))
        print("+=======+======================+======================+\n\n")

    #

    try:
        neutral_state = int(QMin['template']['neutral_state'])
    except:
        neutral_state = QMin.get('calc_neutral')

    if 'calc_neutral' in QMin:
        nactel = int(QMin['template']['nactel']) - 1
        print(
            f"\n===> Neutral state is calculated with converged {QMin['template']['method'].upper()} orbital of the anion\n"
            f"and ({nactel}e,{QMin['template']['ras2']}o) active space.\n")
        if 'grad' in QMin:
            neutral_energy = read_calc_neutral(out)
        else:
            # TODO fix this
            neutral_filename = os.path.join(QMin['scratchdir'], 'NEUTRAL', 'MOLCAS.out')
            neutral_energy = read_calc_neutral(readfile(neutral_filename), gradnacfile=False)
        print(f"  Calculated energy (Hartree): {neutral_energy:14.10f}\n")

    elif 'init' in QMin and 'calc_neutral' not in QMin:
        neutral_energy = H0.diagonal()[neutral_state - 1]
        print("\n===> Neutral state chosen by the user is %i with energy (initial step) (a.u.): %16.12f\n" % (
        neutral_state, neutral_energy))
    else:
        try:
            if 'track_wfoverlap' in QMin:
                neutral_state = _neutral_tracking_WfOverlap(QMin)
                neutral_energy = H0.diagonal()[neutral_state - 1]
                print("\n===> Neutral state is %i with energy (a.u.): %16.12f\n" % (neutral_state, neutral_energy))
                _replaceQMtemplate_line("neutral_state", [neutral_state])
            else:
                neutral_state = int(_neutral_tracking(QMin))
                neutral_energy = H0.diagonal()[neutral_state - 1]
                print("\n===> Neutral state is %i with energy (a.u.): %16.12f\n" % (neutral_state, neutral_energy))
                _replaceQMtemplate_line("neutral_state", [neutral_state])
        except Exception as e:
            print(f"\n ====> An error occurred (check QM.err carefully!): \n\t{e}")
            traceback.print_exc()
            print(
                "\nWarning : Neutral tracking failed. Neutral state (%s) is chosen to be the one from last iteration." % (
                    QMin['template']['neutral_state']))
            neutral_energy = H0.diagonal()[neutral_state - 1]
            print("\n===> Neutral state is %i with energy (a.u.): %16.12f\n" % (neutral_state, neutral_energy))

    if 'pcap' in QMin:
        #        writeQMPCAPout(eVs.diagonal(), act_states, H0.diagonal()[neutral_state - 1], QMin)
        # Store eigen values of diag basis (in nmstates X nmstates format).
        QMout['h_diag'] = eVs
        unsorted_h_diag_res_real = np.zeros([nmstates, nmstates])
        unsorted_h_diag_res_imag = np.zeros([nmstates, nmstates])

        for i, state1 in enumerate(act_states):
            state1 -= 1
            unsorted_h_diag_res_real[i, i] = QMout['h_diag'][state1, state1].real
            unsorted_h_diag_res_imag[i, i] = 2.0 * QMout['h_diag'][state1, state1].imag

        _des_idx = np.argsort(unsorted_h_diag_res_real.diagonal())[::-1]

        QMout['h_diag_res_real'] = np.zeros([nmstates, nmstates])
        QMout['h_diag_res_imag'] = np.zeros([nmstates, nmstates])
        QMout['grad_diag_res'] = []
        QMout['nacdr_diag_res'] = [[[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)] for l
                                   in range(nmstates)]

        np.fill_diagonal(QMout['h_diag_res_real'], [unsorted_h_diag_res_real[i, i] for i in _des_idx])
        np.fill_diagonal(QMout['h_diag_res_imag'], [unsorted_h_diag_res_imag[i, i] for i in _des_idx])

        # Write energies to PCAP.out file in QM/ folder
        writeQMPCAPout(QMout['h_diag_res_real'], QMout['h_diag_res_imag'], neutral_energy, QMin)

    # Get gradient and NACDR matrix.
    # Store MCH gradients
    if 'grad' in QMin and 'pcap' in QMin:
        ghost = True if 'ghost' in QMin else False
        molcas_obj = ParseFile_MOLCAS.ParseFile(out, nroots, natom, ghost)
        grad, nac = molcas_obj.grad_mat()

        gradAll = []
        for state in range(nroots):
            gradAll.append([[grad[_atom]['x'][state, state],
                             grad[_atom]['y'][state, state],
                             grad[_atom]['z'][state, state]] for _atom in range(natom)])

        QMout['grad'] = deepcopy(gradAll)

    # Store MCH NACDR (<psi|psi'>) vectors
    if 'nacdr' in QMin and 'pcap' in QMin:
        nacdr = [[[[0. for i in range(3)] for j in range(natom)] for k in range(nroots)] for l in range(nroots)]
        for _state in range(nroots):
            for _state2 in range(_state + 1, nroots):
                nacdr[_state][_state2] = [[nac[_atom]['x'][_state, _state2],
                                           nac[_atom]['y'][_state, _state2],
                                           nac[_atom]['z'][_state, _state2]] for _atom in range(natom)]

                nacdr[_state2][_state] = deepcopy(nacdr[_state][_state2])
                for x in range(natom):
                    for y in range(3):
                        nacdr[_state2][_state][x][y] *= -1.

        QMout['nacdr'] = deepcopy(nacdr)

        # Get Gradient correction coming form CAP
        Grad_correct = {}
        for iatom in range(natom):
            Grad_correct[iatom] = {'x': np.zeros([nroots, nroots], dtype=complex),
                                   'y': np.zeros([nroots, nroots], dtype=complex),
                                   'z': np.zeros([nroots, nroots], dtype=complex)}

        if 'no_capgrad_correct' not in QMin:
            try:
                Grad_correct = CAPG_MAT(outfilename, nroots, QMin, do_numerical=True, saveRASSI=False)
            except Exception as e:
                print("Error in gradient correction calculation. No correction added.\n")
                traceback.print_exc()

        # ===================================Can comment out later!=============================================
        grad_correct_print = []
        for j in range(natom):
            grad_diag_print = {}
            for cart in ['x', 'y', 'z']:
                _grad_MCH = 1.0j * float(QMin['template']['cap_eta']) * (1E-5) * Grad_correct[j][cart]
                _grad_MCH = reduce(np.dot, (QMin['u_xms'].T, _grad_MCH, QMin['u_xms']))
                _diag = reduce(np.dot, (Leigvc.T, _grad_MCH, Reigvc))
                grad_diag_print[cart] = np.diagonal(_diag).real
            grad_correct_print.append({'x': grad_diag_print['x'],
                                       'y': grad_diag_print['y'],
                                       'z': grad_diag_print['z']})

        grad_diag_res_print = []
        for i, state1 in enumerate(act_states):
            state1 -= 1
            grad_diag_res_print.append([[grad_correct_print[_atom]['x'][state1].real,
                                         grad_correct_print[_atom]['y'][state1].real,
                                         grad_correct_print[_atom]['z'][state1].real] for _atom in range(natom)])

        #        float_format = "{:12.8f}"
        print("\n===> CAP contribution to gradient in diagonal basis (Hartree/bohr)")
        for state_index, state_gradients in enumerate(np.asarray(grad_diag_res_print)):
            title = f"Resonance state: {state_index + 1}"
            headers = ['Atom', 'X', 'Y', 'Z']
            data = list(zip([QMin['geo'][i][0] for i in range(QMin['natom'])],
                            state_gradients[:, 0],
                            state_gradients[:, 1],
                            state_gradients[:, 2]))
            print_to_table(title, headers, data, align="right")

        '''
            table_data = []
            for atom_index, atom_gradients in enumerate(state_gradients, start=1):
                formatted_gradients = [float_format.format(_grad) for _grad in atom_gradients]
                table_data.append([atom_index] + formatted_gradients)
            print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        print("\n")
        '''
        del grad_diag_print
        del grad_diag_res_print
        # ===================================Can comment out later!=============================================

        # Calculate Gradient and NACDR matrices in Diag basis
        nacDIAG = {}
        gradDIAG = {}
        for iatom in range(natom):
            nacDIAG[iatom] = {'x': np.zeros([nroots, nroots], dtype=complex),
                              'y': np.zeros([nroots, nroots], dtype=complex),
                              'z': np.zeros([nroots, nroots], dtype=complex)}
        for iatom in range(natom):
            gradDIAG[iatom] = {'x': np.zeros([nroots, nroots], dtype=complex),
                               'y': np.zeros([nroots, nroots], dtype=complex),
                               'z': np.zeros([nroots, nroots], dtype=complex)}

        for j in range(natom):
            # Step1: Create the NAC matrix in ZO basis.
            for s1 in range(nroots):
                for s2 in range(s1 + 1, nroots):
                    for cart in ['x', 'y', 'z']:
                        nac[j][cart][s1, s2] *= np.real(H0[s2, s2] - H0[s1, s1])  # This is NAC <psi|dH/dR|psi>
                        nac[j][cart][s2, s1] = copy.deepcopy(nac[j][cart][s1, s2])

            # Step2: Create the full [G] matrix in ZO basis.
            # Rotate the ZO [G] matrix via Left and right eigenvectors to create [G] in diag basis.
            for cart in ['x', 'y', 'z']:
                _grad_MCH = nac[j][cart] + grad[j][cart] + \
                            1.0j * float(QMin['template']['cap_eta']) * (1E-5) * Grad_correct[j][
                                cart]  # Adding gradient corrections.
                _grad_MCH = reduce(np.dot, (QMin['u_xms'].T, _grad_MCH, QMin['u_xms']))
                _diag = reduce(np.dot, (Leigvc.T, _grad_MCH, Reigvc))
                nacDIAG[j][cart] = copy.deepcopy(_diag)
                np.fill_diagonal(nacDIAG[j][cart], 0.0 + 0.0j)  # No diagonal elements for NAC matrix
                gradDIAG[j][cart] = copy.deepcopy(np.diagonal(_diag))
                # print_complex_matrix(nacDIAG[j][cart], header="NACDR: (Atom %s) '%s\n"%(j, cart))

            # Step3: Create NACDR matrix from NAC matrix
            for s1 in range(nroots):
                for s2 in range(s1 + 1, nroots):
                    for cart in ['x', 'y', 'z']:
                        _eDiff = (QMout['h_diag'][s2, s2] - QMout['h_diag'][s1, s1])  # Taking the complex energy
                        # print(_eDiff)
                        try:
                            # _tmp=deepcopy(nacDIAG[j][cart][s1, s2])
                            nacDIAG[j][cart][s1, s2] /= _eDiff
                            nacDIAG[j][cart][s2, s1] /= (-1.0 * _eDiff)
                            # print("%s-->%s"%(s1+1, s2+1), _eDiff, _tmp,  nacDIAG[j][cart][s1, s2])
                        except:
                            print("Degeneracy detected (Diag basis) (%s & %s)" % (s1 + 1, s2 + 1))
                            nacDIAG[j][cart][s1, s2] /= complex(1E-8, 0.0)  # Singularity check
                            nacDIAG[j][cart][s2, s1] /= complex(1E-8, 0.0)  # Singularity check

        # Store NACDR and Gradient matrix in diag basis.
        if 'nacdr' in QMin:
            nacdr = [[[[0. for i in range(3)] for j in range(natom)] for k in range(nroots)] for l in range(nroots)]
            for _state in range(nroots):
                for _state2 in range(_state + 1, nroots):
                    nacdr[_state][_state2] = [[nacDIAG[_atom]['x'][_state][_state2],
                                               nacDIAG[_atom]['y'][_state][_state2],
                                               nacDIAG[_atom]['z'][_state][_state2]] for _atom in range(natom)]
                    nacdr[_state2][_state] = [[nacDIAG[_atom]['x'][_state2][_state],
                                               nacDIAG[_atom]['y'][_state2][_state],
                                               nacDIAG[_atom]['z'][_state2][_state]] for _atom in range(natom)]
                    # NACdr pairs are not equal in magnitude cause of biorthogonalized set of eigen vectors.

        QMout['nacdr_diag'] = deepcopy(nacdr)

        if 'grad' in QMin:
            gradAll = []
            for state in range(nroots):
                gradAll.append([[gradDIAG[_atom]['x'][state],
                                 gradDIAG[_atom]['y'][state],
                                 gradDIAG[_atom]['z'][state]] for _atom in range(natom)])

        QMout['grad_diag'] = deepcopy(gradAll)
        # Now choose the active states' properties
        unsorted_grad_diag_res = []
        unsorted_nacdr_diag_res = [[[[0. for i in range(3)] for j in range(natom)] for k in range(nmstates)] for l in
                                   range(nmstates)]

        for i, state1 in enumerate(act_states):
            state1 -= 1
            unsorted_grad_diag_res.append([[gradDIAG[_atom]['x'][state1].real,
                                            gradDIAG[_atom]['y'][state1].real,
                                            gradDIAG[_atom]['z'][state1].real] for _atom in range(natom)])
            for j in range(i + 1, len(act_states)):
                state2 = act_states[j] - 1
                unsorted_nacdr_diag_res[j][i] = [[nacDIAG[_atom]['x'][state2][state1].real,
                                                  nacDIAG[_atom]['y'][state2][state1].real,
                                                  nacDIAG[_atom]['z'][state2][state1].real] for _atom in range(natom)]
                unsorted_nacdr_diag_res[i][j] = [[nacDIAG[_atom]['x'][state1][state2].real,
                                                  nacDIAG[_atom]['y'][state1][state2].real,
                                                  nacDIAG[_atom]['z'][state1][state2].real] for _atom in range(natom)]
                # They are not symmetric

        # Rearrange the numbers to upper and lower surfaces.
        if 'pcap' in QMin:
            _des_idx = np.argsort(unsorted_h_diag_res_real.diagonal())[::-1]

            QMout['grad_diag_res'] = [unsorted_grad_diag_res[i] for i in _des_idx]
            for _idxI, _ in enumerate(_des_idx):
                for _idxJ in range(_idxI + 1, len(act_states)):
                    QMout['nacdr_diag_res'][_idxI][_idxJ] = unsorted_nacdr_diag_res[_des_idx[_idxI]][_des_idx[_idxJ]]
                    QMout['nacdr_diag_res'][_idxJ][_idxI] = unsorted_nacdr_diag_res[_des_idx[_idxJ]][_des_idx[_idxI]]

        if 'pcap' in QMin:  # TOFIX: SUCH a shoddy solution.
            QMout['dm'] = [[[complex(0.0) for i in range(nmstates)] for j in range(nmstates)] for xyz in range(3)]

    if 'overlap' in QMin:
        nac = makecmatrix(nmstates, nmstates)
        for istate, i in enumerate(QMin['statemap']):
            for jstate, j in enumerate(QMin['statemap']):
                mult1, state1, ms1 = tuple(QMin['statemap'][i])
                mult2, state2, ms2 = tuple(QMin['statemap'][j])
                if mult1 == mult2 and ms1 == ms2:
                    nac[istate][jstate] = complex(getsmate(out, mult1, state1, state2, states))
                else:
                    nac[istate][jstate] = complex(0.0)
        QMout['overlap'] = nac
    # Phases from overlaps
    if 'phases' in QMin:
        if 'phases' not in QMout:
            QMout['phases'] = [complex(1., 0.) for i in range(nmstates)]
        if 'overlap' in QMout:
            for i in range(nmstates):
                if QMout['overlap'][i][i].real < 0.:
                    QMout['phases'][i] = complex(-1., 0.)
    return QMout


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #
def writeQMout(QMin, QMout, QMinfilename):
    '''Writes the requested quantities to the file which SHARC reads in. The filename is QMinfilename with everything after the first dot replaced by "out".

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout
    3 string: QMinfilename'''

    k = QMinfilename.find('.')
    if k == -1:
        outfilename = QMinfilename + '.out'
    else:
        outfilename = QMinfilename[:k] + '.out'
    if PRINT:
        print('===> Writing output to file %s in SHARC Format\n' % (outfilename))
    string = ''
    if 'h' in QMin or 'soc' in QMin:
        string += writeQMoutsoc(QMin, QMout)
    if 'dm' in QMin:
        string += writeQMoutdm(QMin, QMout)
    if 'grad' in QMin:
        string += writeQMoutgrad(QMin, QMout)
    if 'nacdr' in QMin:
        string += writeQMoutnacana(QMin, QMout)
    if 'overlap' in QMin:
        string += writeQMoutnacsmat(QMin, QMout)
    if 'phases' in QMin:
        string += writeQmoutPhases(QMin, QMout)
    string += writeQMouttime(QMin, QMout)
    outfile = os.path.join(QMin['pwd'], outfilename)
    writefile(outfile, string)
    return


# ======================================================================= #

def writeQMPCAPout(energy_real, energy_imag, neutral, QMin):
    PCAPfilename = 'PCAP.out'
    string = ''
    if 'init' in QMin:
        string += '#'
        string += ' '.join([str("\tNeutral in a.u.\t")])
        string += ' '.join([str("\tRe(E%i) in a.u.\t" % (j + 1)) for j in range(len(energy_real))])
        string += ' '.join([str("\tGamma(E%i) in a.u.\t" % (j + 1)) for j in range(len(energy_real))])
        string += '\n'

    string += ' '.join([str("   %16.12E\t" % neutral)])
    string += ' '.join([str("   %16.12E\t" % energy_real[j, j]) for j in range(len(energy_real))])
    string += ' '.join([str("   %16.12E\t" % energy_imag[j, j]) for j in range(len(energy_real))])
    string += '\n'

    with open(PCAPfilename, 'a') as file:
        file.write(string)

    return


def writeQMoutsoc(QMin, QMout):
    '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the SOC matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['h'][i][j].real, 9, 3), eformat(QMout['h'][i][j].imag, 9, 3))
        string += '\n'
    string += '\n'
    return string


# ======================================================================= #


def writeQMoutdm(QMin, QMout):
    '''Generates a string with the Dipole moment matrices in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The string contains three such matrices.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the DM matrices'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
    for xyz in range(3):
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(nmstates):
            for j in range(nmstates):
                string += '%s %s ' % (
                    eformat(QMout['dm'][xyz][i][j].real, 9, 3), eformat(QMout['dm'][xyz][i][j].imag, 9, 3))
            string += '\n'
        # string+='\n'
    return string


# ======================================================================= #


def writeQMoutgrad(QMin, QMout):
    '''Generates a string with the Gradient vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates gradients are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the Gradient vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
        for atom in range(natom):
            for xyz in range(3):
                string += '%s ' % (eformat(QMout['grad'][i][atom][xyz], 9, 3))
            string += '\n'
        # string+='\n'
        i += 1
    return string


# ======================================================================= #


def writeQMoutnacana(QMin, QMout):
    '''Generates a string with the NAC vectors in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (nmstates x nmstates vectors are written).

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the NAC vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            string += '%i %i ! %i %i %i %i %i %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms)
            for atom in range(natom):
                for xyz in range(3):
                    string += '%s ' % (eformat(QMout['nacdr'][i][j][atom][xyz], 12, 3))
                string += '\n'
            string += ''
            j += 1
        i += 1
    return string


# ======================================================================= #


def writeQMoutnacsmat(QMin, QMout):
    '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the transformation matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for j in range(nmstates):
        for i in range(nmstates):
            string += '%s %s ' % (
                eformat(QMout['overlap'][j][i].real, 9, 3), eformat(QMout['overlap'][j][i].imag, 9, 3))
        string += '\n'
    string += '\n'
    return string


# ======================================================================= #
def writeQMouttime(QMin, QMout):
    '''Generates a string with the quantum mechanics total runtime in SHARC format.

    The string starts with a ! followed by a flag specifying the type of data. In the next line, the runtime is given

    Arguments:
    1 dictionary: QMin
    2 dictionary: QMout

    Returns:
    1 string: multiline string with the runtime'''

    string = '! 8 Runtime\n%s\n' % (eformat(QMout['runtime'], 9, 3))
    return string


# ======================================================================= #
def writeQmoutPhases(QMin, QMout):
    string = '! 7 Phases\n%i ! for all nmstates\n' % (QMin['nmstates'])
    for i in range(QMin['nmstates']):
        string += '%s %s\n' % (eformat(QMout['phases'][i].real, 9, 3), eformat(QMout['phases'][i].imag, 9, 3))
    return string


# =============================================================================================== #
# =============================================================================================== #
# =========================================== SUBROUTINES TO readQMin =========================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def checkscratch(SCRATCHDIR):
    '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

    Arguments:
    1 string: path to SCRATCHDIR'''

    exist = os.path.exists(SCRATCHDIR)
    if exist:
        isfile = os.path.isfile(SCRATCHDIR)
        if isfile:
            print('$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR))
            sys.exit(28)
    else:
        try:
            os.makedirs(SCRATCHDIR)
        except OSError:
            print('Cannot create SCRATCHDIR=%s\n' % (SCRATCHDIR))
            sys.exit(29)


# ======================================================================= #


def removequotes(string):
    if string.startswith("'") and string.endswith("'"):
        return string[1:-1]
    elif string.startswith('"') and string.endswith('"'):
        return string[1:-1]
    else:
        return string


# ======================================================================= #


def getsh2caskey(sh2cas, key):
    i = -1
    while True:
        i += 1
        try:
            line = re.sub('#.*$', '', sh2cas[i])
        except IndexError:
            break
        line = line.split(None, 1)
        if line == []:
            continue
        if key.lower() in line[0].lower():
            return line
    return ['', '']


# ======================================================================= #


def get_sh2cas_environ(sh2cas, key, environ=True, crucial=True):
    line = getsh2caskey(sh2cas, key)
    if line[0]:
        LINE = line[1]
        LINE = removequotes(LINE).strip()
    else:
        if environ:
            LINE = os.getenv(key.upper())
            if not LINE:
                if crucial:
                    print('Either set $%s or give path to %s in MOLCAS.resources!' % (key.upper(), key.upper()))
                    sys.exit(30)
                else:
                    return ''
        else:
            if crucial:
                print('Give path to %s in MOLCAS.resources!' % (key.upper()))
                sys.exit(31)
            else:
                return ''
    LINE = os.path.expandvars(LINE)
    LINE = os.path.expanduser(LINE)
    if containsstring(';', LINE):
        print(
            "$%s contains a semicolon. Do you probably want to execute another command after %s? I can't do that for you..." % (
                key.upper(), key.upper()))
        sys.exit(32)
    return LINE


# ======================================================================= #


def get_pairs(QMinlines, i):
    nacpairs = []
    while True:
        i += 1
        try:
            line = QMinlines[i].lower()
        except IndexError:
            print('"keyword select" has to be completed with an "end" on another line!')
            sys.exit(33)
        if 'end' in line:
            break
        fields = line.split()
        try:
            nacpairs.append([int(fields[0]), int(fields[1])])
        except ValueError:
            print('"nacdr select" is followed by pairs of state indices, each pair on a new line!')
            sys.exit(34)
    return nacpairs, i


# ======================================================================= #         OK


def readQMin(QMinfilename):
    '''Reads the time-step dependent information from QMinfilename. This file contains all information from the current SHARC job: geometry, velocity, number of states, requested quantities along with additional information. The routine also checks this input and obtains a number of environment variables necessary to run MOLCAS.

    Steps are:
    - open and read QMinfilename
    - Obtain natom, comment, geometry (, velocity)
    - parse remaining keywords from QMinfile
    - check keywords for consistency, calculate nstates, nmstates
    - obtain environment variables for path to MOLCAS and scratch directory, and for error handling

    Arguments:
    1 string: name of the QMin file

    Returns:
    1 dictionary: QMin'''

    # read QMinfile
    QMinlines = readfile(QMinfilename)
    QMin = {}

    # Get natom
    try:
        natom = int(QMinlines[0])
    except ValueError:
        print('first line must contain the number of atoms!')
        sys.exit(35)
    QMin['natom'] = natom
    if len(QMinlines) < natom + 4:
        print('Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task')
        sys.exit(36)

    # Save Comment line
    QMin['comment'] = QMinlines[1]

    # Get geometry and possibly velocity (for backup-analytical non-adiabatic couplings)
    QMin['geo'] = []
    QMin['veloc'] = []
    hasveloc = True
    for i in range(2, natom + 2):
        if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*',
                              QMinlines[i]):
            print('Input file does not comply to xyz file format! Maybe natom is just wrong.')
            sys.exit(37)
        fields = QMinlines[i].split()
        for j in range(1, 4):
            fields[j] = float(fields[j])
        QMin['geo'].append(fields[0:4])
        if len(fields) >= 7:
            for j in range(4, 7):
                fields[j] = float(fields[j])
            QMin['veloc'].append(fields[4:7])
        else:
            hasveloc = False
    if not hasveloc:
        QMin = removekey(QMin, 'veloc')

    # Parse remaining file
    i = natom + 1
    while i + 1 < len(QMinlines):
        i += 1
        line = QMinlines[i]
        line = re.sub('#.*$', '', line)
        if len(line.split()) == 0:
            continue
        key = line.lower().split()[0]
        if 'savedir' in key:
            args = line.split()[1:]
        else:
            args = line.lower().split()[1:]
        if key in QMin:
            print('Repeated keyword %s in line %i in input file! Check your input!' % (key, i + 1))
            continue  # only first instance of key in QM.in takes effect
        if len(args) >= 1 and 'select' in args[0]:
            pairs, i = get_pairs(QMinlines, i)
            QMin[key] = pairs
        else:
            QMin[key] = args

    if 'unit' in QMin:
        if QMin['unit'][0] == 'angstrom':
            factor = 1. / au2a
        elif QMin['unit'][0] == 'bohr':
            factor = 1.
        else:
            print('Dont know input unit %s!' % (QMin['unit'][0]))
            sys.exit(38)
    else:
        factor = 1. / au2a

    for iatom in range(len(QMin['geo'])):
        for ixyz in range(3):
            QMin['geo'][iatom][ixyz + 1] *= factor

    if 'states' not in QMin:
        print('Keyword "states" not given!')
        sys.exit(39)
    # Calculate states, nstates, nmstates
    for i in range(len(QMin['states'])):
        QMin['states'][i] = int(QMin['states'][i])
    reduc = 0
    for i in reversed(QMin['states']):
        if i == 0:
            reduc += 1
        else:
            break
    for i in range(reduc):
        del QMin['states'][-1]
    nstates = 0
    nmstates = 0
    for i in range(len(QMin['states'])):
        nstates += QMin['states'][i]
        nmstates += QMin['states'][i] * (i + 1)
    QMin['nstates'] = nstates
    QMin['nmstates'] = nmstates

    # Various logical checks
    if 'states' not in QMin:
        print('Number of states not given in QM input file %s!' % (QMinfilename))
        sys.exit(40)

    possibletasks = ['h', 'dm', 'grad', 'overlap', 'phases', 'PCAP']  # Added new task projected-task pCAP
    if not any([i in QMin for i in possibletasks]):
        print('No tasks found! Tasks are "h", "dm", "grad", "overlap".')
        sys.exit(41)

    if 'samestep' in QMin and 'init' in QMin:
        print('"Init" and "Samestep" cannot be both present in QM.in!')
        sys.exit(42)

    if 'phases' in QMin:
        if 'pcap' not in QMin:  # SBK Added this trap here
            QMin['overlap'] = []

    if 'overlap' in QMin and 'init' in QMin:
        print('"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"')
        sys.exit(43)

    if 'init' not in QMin and 'samestep' not in QMin:
        QMin['newstep'] = []

#    if not any([i in QMin for i in ['h', 'dm', 'grad']]) and 'overlap' in QMin:

    QMin['h'] = []

    if len(QMin['states']) > 8:
        print('Higher multiplicities than octets are not supported!')
        sys.exit(44)

    if 'h' in QMin and 'soc' in QMin:
        QMin = removekey(QMin, 'h')

    if 'nacdt' in QMin:
        print('Within the SHARC-MOLCAS interface, "nacdt" is not supported.')
        sys.exit(45)

    if 'molden' in QMin:
        os.environ['MOLCAS_MOLDEN'] = 'ON'
        if 'samestep' in QMin:
            print('HINT: Not producing Molden files in "samestep" mode!')
            del QMin['molden']

    # Check for correct gradient list
    if 'grad' in QMin:
        if len(QMin['grad']) == 0 or QMin['grad'][0] == 'all':
            QMin['grad'] = [i + 1 for i in range(nmstates)]
            # pass
        else:
            for i in range(len(QMin['grad'])):
                try:
                    QMin['grad'][i] = int(QMin['grad'][i])
                except ValueError:
                    print('Arguments to keyword "grad" must be "all" or a list of integers!')
                    sys.exit(47)
                if QMin['grad'][i] > nmstates:
                    print('State for requested gradient does not correspond to any state in QM input file state list!')
                    sys.exit(48)

    # Process the overlap requests
    # identically to the nac requests
    if 'overlap' in QMin:
        if len(QMin['overlap']) >= 1:
            nacpairs = QMin['overlap']
            for i in range(len(nacpairs)):
                if nacpairs[i][0] > nmstates or nacpairs[i][1] > nmstates:
                    print(
                        'State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!')
                    sys.exit(49)
        else:
            QMin['overlap'] = [[j + 1, i + 1] for i in range(nmstates) for j in range(i + 1)]

    # Process the non-adiabatic coupling requests
    # type conversion has already been done
    if 'nacdr' in QMin:
        if len(QMin['nacdr']) >= 1:
            nacpairs = QMin['nacdr']
            for i in range(len(nacpairs)):
                if nacpairs[i][0] > nmstates or nacpairs[i][1] > nmstates:
                    print(
                        'State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!')
                    sys.exit(50)
        else:
            QMin['nacdr'] = [[j + 1, i + 1] for i in range(nmstates) for j in range(i)]

    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(QMin['states']):
        statemap[i] = [imult, istate, ims]
        i += 1
    QMin['statemap'] = statemap

    # get the set of states for which gradients actually need to be calculated
    gradmap = set()
    if 'grad' in QMin:
        for i in QMin['grad']:
            gradmap.add(tuple(statemap[i][0:2]))
    gradmap = sorted(gradmap)
    QMin['gradmap'] = gradmap

    # get the list of statepairs for NACdr calculation
    nacmap = set()
    if 'nacdr' in QMin:
        for i in QMin['nacdr']:
            s1 = statemap[i[0]][0:2]
            s2 = statemap[i[1]][0:2]
            if s1[0] != s2[0] or s1 == s2:
                continue
            if s1[1] > s2[1]:
                continue
            nacmap.add(tuple(s1 + s2))
    nacmap = list(nacmap)
    nacmap.sort()
    QMin['nacmap'] = nacmap

    # open MOLCAS.resources
    filename = 'MOLCAS.resources'
    if os.path.isfile(filename):
        sh2cas = readfile(filename)
    else:
        print('Warning: No MOLCAS.resources found!')
        print('Reading resources from SH2CAS.inp')
        sh2cas = readfile('SH2CAS.inp')

    QMin['pwd'] = os.getcwd()

    QMin['molcas'] = get_sh2cas_environ(sh2cas, 'molcas')
    os.environ['MOLCAS'] = QMin['molcas']

    # SBK added
    try:
        QMin['ncpu_avail'] = get_sh2cas_environ(sh2cas, 'ncpu_avail')
    except:
        QMin['ncpu_avail'] = 8

    driver = get_sh2cas_environ(sh2cas, 'driver', crucial=False)
    if driver == '':
        driver = os.path.join(QMin['molcas'], 'bin', 'pymolcas')
        if not os.path.isfile(driver):
            driver = os.path.join(QMin['molcas'], 'bin', 'molcas.exe')
            if not os.path.isfile(driver):
                print(
                    'No driver (pymolcas or molcas.exe) found in $MOLCAS/bin. Please add the path to the driver via the "driver" keyword.')
                sys.exit(52)
    QMin['driver'] = driver

    if 'pcap' in QMin:
        # Even though wfoverlap is not used, this is called!
        QMin['wfoverlap'] = get_sh2cas_environ(sh2cas, 'wfoverlap', crucial=False)
        if not QMin['wfoverlap']:
            ciopath = os.path.join(os.path.expandvars(os.path.expanduser('$SHARC')), 'wfoverlap.x')
            if os.path.isfile(ciopath):
                QMin['wfoverlap'] = ciopath
            else:
                print('Give path to wfoverlap.x in MOLCAS.resources!')
                sys.exit(51)

    # Set up scratchdir
    line = get_sh2cas_environ(sh2cas, 'scratchdir', False, False)
    if line is None:
        line = QMin['pwd'] + '/SCRATCHDIR/'
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)

    QMin['scratchdir'] = line

    # Set up savedir
    if 'savedir' in QMin:
        line = QMin['savedir'][0]
    else:
        line = get_sh2cas_environ(sh2cas, 'savedir', False, False)
        if line is None or line == '':
            line = os.path.join(QMin['pwd'], 'SAVEDIR')
    line = os.path.expandvars(line)
    line = os.path.expanduser(line)
    line = os.path.abspath(line)
    if 'init' in QMin:
        checkscratch(line)
    QMin['savedir'] = line

    line = getsh2caskey(sh2cas, 'debug')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            global DEBUG
            DEBUG = True

    line = getsh2caskey(sh2cas, 'no_print')
    if line[0]:
        if len(line) <= 1 or 'true' in line[1].lower():
            global PRINT
            PRINT = False

    QMin['memory'] = 8000
    # SBK changed here
    line = getsh2caskey(sh2cas, 'memory')
    if line[0]:
        try:
            QMin['memory'] = int(line[1])
        except ValueError:
            print('MOLCAS memory does not evaluate to numerical value!')
            sys.exit(54)
    else:
        print(
            'WARNING: Please set memory for MOLCAS in MOLCAS.resources (in MB)! Using 8000 MB default value!')  # SBK changed here
    os.environ['MOLCASMEM'] = str(QMin['memory'])
    os.environ['MOLCAS_MEM'] = str(QMin['memory'])

    QMin['ncpu'] = 1
    line = getsh2caskey(sh2cas, 'ncpu')
    if line[0]:
        try:
            QMin['ncpu'] = int(line[1])
        except ValueError:
            print('Number of CPUs does not evaluate to numerical value!')
            sys.exit(55)

    QMin['mpi_parallel'] = False
    line = getsh2caskey(sh2cas, 'mpi_parallel')
    if line[0]:
        QMin['mpi_parallel'] = True

    QMin['schedule_scaling'] = 0.6
    line = getsh2caskey(sh2cas, 'schedule_scaling')
    if line[0]:
        try:
            x = float(line[1])
            if 0 < x <= 2.:
                QMin['schedule_scaling'] = x
        except ValueError:
            print('Schedule scaling does not evaluate to numerical value!')
            sys.exit(56)

    QMin['Project'] = 'MOLCAS'
    os.environ['Project'] = QMin['Project']

    QMin['delay'] = 0.0
    line = getsh2caskey(sh2cas, 'delay')
    if line[0]:
        try:
            QMin['delay'] = float(line[1])
        except ValueError:
            print('Submit delay does not evaluate to numerical value!')
            sys.exit(57)

    QMin['Project'] = 'MOLCAS'
    os.environ['Project'] = QMin['Project']
    os.environ['MOLCAS_OUTPUT'] = 'PWD'

    line = getsh2caskey(sh2cas, 'always_orb_init')
    if line[0]:
        QMin['always_orb_init'] = []
    line = getsh2caskey(sh2cas, 'always_guess')
    if line[0]:
        QMin['always_guess'] = []
    # SBK adding a guess word scforb
    line = getsh2caskey(sh2cas, 'scforb')
    if line[0]:
        QMin['scforb'] = []
    #
    # SBK adding a guess word rassorb
    line = getsh2caskey(sh2cas, 'rasorb')
    if line[0]:
        QMin['rasorb'] = []
    #
    # SBK adds a key word save_integral
    line = getsh2caskey(sh2cas, 'save_integral')
    if line[0]:
        QMin['save_integral'] = True
    else:
        QMin['save_integral'] = False

    # SBK changed the if statement:
    conditions = [
        'always_orb_init' in QMin,
        'always_guess' in QMin,
        "scforb" in QMin
    ]

    if sum(conditions) >= 2:
        print('Any two of these keywords : "always_orb_init","always_guess" and "scforb" cannot be used together!')
        sys.exit(58)

    # open template
    template = readfile('MOLCAS.template')

    QMin['template'] = {}
    integers = ['nactel', 'inactive', 'ras2', 'frozen']
    cap_params = ['cap_eta', 'cap_x', 'cap_y', 'cap_z', 'screening', 'r_cut']
    strings = ['basis', 'method', 'baslib', 'pdft-functional']
    floats = ['ipea', 'imaginary', 'gradaccumax', 'gradaccudefault', 'displ', 'rasscf_thrs_e', 'rasscf_thrs_rot',
              'rasscf_thrs_egrd', 'cholesky_accu']
    booleans = ['cholesky', 'no-douglas-kroll', 'douglas-kroll', 'cholesky_analytical']
    for i in booleans:
        QMin['template'][i] = False
    QMin['template']['roots'] = [0 for i in range(8)]
    QMin['template']['rootpad'] = [0 for i in range(8)]
    QMin['template']['method'] = 'casscf'
    QMin['template']['pdft-functional'] = 't:pbe'
    QMin['template']['baslib'] = ''
    QMin['template']['ipea'] = 0.25
    QMin['template']['imaginary'] = 0.00
    QMin['template']['frozen'] = -1
    QMin['template']['iterations'] = [200, 100]
    QMin['template']['gradaccumax'] = 1.e-2
    QMin['template']['gradaccudefault'] = 1.e-4
    QMin['template']['displ'] = 0.005
    QMin['template']['cholesky_accu'] = 1e-4
    QMin['template']['rasscf_thrs_e'] = 1e-10
    QMin['template']['rasscf_thrs_rot'] = 1e-6
    QMin['template']['rasscf_thrs_egrd'] = 1e-6
    QMin['template']['no-douglas-kroll'] = True
    QMin['template']['charge'] = -1

    for line in template:
        orig = re.sub('#.*$', '', line).split(None, 1)
        line = re.sub('#.*$', '', line).lower().split()
        if len(line) == 0:
            continue
        if 'spin' in line[0]:
            QMin['template']['roots'][int(line[1]) - 1] = int(line[3])
        elif 'roots' in line[0]:
            for i, n in enumerate(line[1:]):
                QMin['template']['roots'][i] = int(n)
        elif 'rootpad' in line[0]:
            for i, n in enumerate(line[1:]):
                QMin['template']['rootpad'][i] = int(n)
        elif 'baslib' in line[0]:
            QMin['template']['baslib'] = os.path.abspath(orig[1])
        elif 'iterations' in line[0]:
            if len(line) >= 3:
                QMin['template']['iterations'] = [int(i) for i in line[-2:]]
            elif len(line) == 2:
                QMin['template']['iterations'][0] = int(line[-1])
        elif line[0] in integers:
            QMin['template'][line[0]] = int(line[1])
        elif line[0] in booleans:
            QMin['template'][line[0]] = True
        elif line[0] in strings:
            QMin['template'][line[0]] = line[1]
        elif line[0] in floats:
            QMin['template'][line[0]] = float(line[1])
        elif line[0] in cap_params:
            QMin['template'][line[0]] = float(line[1])
        # SBK added this, careful, "act_states" starts at 1 not at 0.
        elif 'act_states' in line[0]:
            QMin['template'][line[0]] = [int(i) for i in line[1:]]
            # SBK: act_states number should be equal to nstates
            if len(QMin['template']['act_states']) != sum(QMin['states']):
                print(
                    "No of states %s not equal to no. of resonance states (%s) chosen. Change 'act_states' or 'states'" % (
                        QMin['template']['act_states'], sum(QMin['states'])))
                sys.exit(911)  # SBK added
        elif 'neutral_state' in line[0]:
            QMin['template'][line[0]] = int(line[1])  # starts at 1 not at 0.
        elif 'calc_neutral' in line[0]:
            QMin[line[0]] = []
        elif 'track_all' in line[0]:
            QMin['track_all'] = []
            # TRACK all states by RASSI overlap, expensive, not a default choice!
            try:
                if 'adapt' in line[1]:
                    QMin['adaptive_track'] = []
            except:
                pass
        elif 'update_ref' in line[0]:
            QMin['template']['update_ref'] = []
            for _ct, _ in enumerate(QMin['template']['act_states']):
                if line[_ct + 1].lower() == 'false':
                    QMin['template']['update_ref'].append(False)
                elif line[_ct + 1].lower() == 'true':
                    QMin['template']['update_ref'].append(True)

        elif 'integral_link' in line[0]:
            # To save a lot of space, OneInt, OrdInt, RunFile will be symlinked from 'savedir' to WORKDIRs,
            # may be slow cause of overheading!
            QMin['integral_link'] = []
        elif 'track_wfoverlap' in line[0]:
            QMin['track_wfoverlap'] = []
        elif 'natorb' in line[0]:
            QMin['natorb'] = []
        elif 'no_capgrad_correct' in line[0]:
            QMin['no_capgrad_correct'] = []
        elif 'ghost' in line[0]:
            QMin['ghost'] = line[1]
        elif 'xms_correction' in line[0]:
            QMin['xms_correction'] = []

    # CAP parameters, strike out redundant part if needed
    if QMin['template'].get('r_cut'):
        QMin['cap_type'] = 'voronoi' # Default?
        for key in ['cap_x', 'cap_y', 'cap_z']:
            QMin['template'].pop(key, None)
    else:
        QMin['cap_type'] = 'box'

            # SBK: choose any one of the tracking algorithm
    if 'track_all' in QMin and 'track_wfoverlap' in QMin:
        print('Both of these two keywords : "track_wfoverlap","track_all" cannot be used together!')
        sys.exit(581)

    if 'neutral_state' not in QMin['template'] and 'calc_neutral' not in QMin:
        print('Neutral state not found in %s. Taking lowest root as neutral.' % ('MOLCAS.template'))
        QMin['template']['neutral_state'] = 1

    # roots must be larger or equal to states
    for i, n in enumerate(QMin['template']['roots']):
        if i == len(QMin['states']):
            break
        if not n >= QMin['states'][i]:
            print('Too few states in state-averaging in multiplicity %i! %i requested, but only %i given' % (
                i + 1, QMin['states'][i], n))
            sys.exit(59)

    # check rootpad
    for i, n in enumerate(QMin['template']['rootpad']):
        if i == len(QMin['states']):
            break
        if not n >= 0:
            print('Rootpad must not be negative!')
            sys.exit(60)

    # condense roots list
    for i in range(len(QMin['template']['roots']) - 1, 0, -1):
        if QMin['template']['roots'][i] == 0:
            QMin['template']['roots'].pop(i)
        else:
            break
    QMin['template']['rootpad'] = QMin['template']['rootpad'][:len(QMin['template']['roots'])]

    necessary = ['basis', 'nactel', 'ras2', 'inactive']
    for i in necessary:
        if i not in QMin['template']:
            print('Key %s missing in template file!' % (i))
            sys.exit(62)

    # modern OpenMolcas can do analytical Cholesky gradients/NACs
    QMin['template']['cholesky_analytical'] = QMin['template']['cholesky']

    # Douglas-Kroll new treatment
    QMin['template']['no-douglas-kroll'] = not QMin['template']['douglas-kroll']

    # find method
    allowed_methods = ['casscf', 'caspt2', 'ms-caspt2', 'mc-pdft', 'xms-pdft', 'cms-pdft']  # Good for us SBK
    # 0: casscf
    # 1: caspt2 (single state)
    # 2: ms-caspt2
    # 3: mc-pdft (single state)
    # 4: xms-pdft
    # 5: cms-pdft
    for i, m in enumerate(allowed_methods):
        if QMin['template']['method'] == m:
            QMin['method'] = i
            break
    else:
        print('Unknown method "%s" given in MOLCAS.template' % (QMin['template']['method']))
        sys.exit(64)

    # find functional if it is cms-pdft
    if QMin['method'] == 5:
        allowed_functionals = ['tpbe', 't:pbe', 'ft:pbe', 't:blyp', 'ft:blyp', 't:revPBE', 'ft:revPBE', 't:LSDA',
                               'ft:LSDA']
        for i, m in enumerate(allowed_functionals):
            if QMin['template']['pdft-functional'] == m:
                QMin['pdft-functional'] = i
                break
        else:
            print('Warning! No analytical gradients for cms-pdft and "%s" given in MOLCAS.template' % (
                QMin['template']['pdft-functional']))
            print(
                'Using numerical gradients. Analytical gradients only for t:pbe, ft:pbe, t:blyp, ft:blyp, t:revPBE, ft:revPBE, t:LSDA, or ft:LSDA.')
            QMin['pdft-functional'] = -1

    # decide which type of gradients to do:
    # 0 = analytical CASSCF gradients in one MOLCAS input file (less overhead, but recommended only under certain circumstances)
    # 1 = analytical CASSCF gradients in separate MOLCAS inputs, possibly distributed over several CPUs (DEFAULT)
    # 2 = numerical gradients (CASPT2, MS-CASPT2, Cholesky-CASSCF; or for dmdr and socdr), possibly distributed over several CPUs
    # 3 = analytical CMS-PDFT gradients in one MOLCAS input file (less overhead, but recommended only under certain circumstances)
    QMin['ncpu'] = max(1, QMin['ncpu'])

    if 'pcap' in QMin:
        QMin['gradmode'] = 1  # THIS IS NEEDED (06/12)

    # SBK needs to change this part ^^

    # SBK needs the following to get the JobIph file
    # Check the save directory
    if 'pcap' not in QMin:
        try:
            ls = os.listdir(QMin['savedir'])
            err = 0
        except OSError:
            print('Problems reading SCRADIR=%s' % (QMin['savedir']))
            sys.exit(68)
        if 'init' in QMin:
            err = 0
        elif 'samestep' in QMin:
            for imult, nstates in enumerate(QMin['states']):
                if nstates < 1:
                    continue
                if not 'MOLCAS.%i.JobIph' % (imult + 1) in ls:
                    print('File "MOLCAS.%i.JobIph" missing in SAVEDIR!' % (imult + 1))
                    err += 1
            if 'overlap' in QMin:
                for imult, nstates in enumerate(QMin['states']):
                    if nstates < 1:
                        continue
                    if not 'MOLCAS.%i.JobIph.old' % (imult + 1) in ls:
                        print('File "MOLCAS.%i.JobIph.old" missing in SAVEDIR!' % (imult + 1))
                        err += 1
        elif 'overlap' in QMin:
            for imult, nstates in enumerate(QMin['states']):
                if nstates < 1:
                    continue
                if not 'MOLCAS.%i.JobIph' % (imult + 1) in ls:
                    print('File "MOLCAS.%i.JobIph" missing in SAVEDIR!' % (imult + 1))
                    err += 1
        if err > 0:
            print('%i files missing in SAVEDIR=%s' % (err, QMin['savedir']))
            sys.exit(69)
    else:
        print("\n===> Continuing OPENCAP-MD trajectory calculation!\n\n")
        # TODO SBK needs to put failsafe options
    QMin['version'] = getversion([''] * 50, QMin['molcas'])

    if PRINT:
        printQMin(QMin)

    return QMin


# =============================================================================================== #
# =============================================================================================== #
# =========================================== gettasks and setup routines ======================= #
# =============================================================================================== #
# =============================================================================================== #

def gettasks(QMin):
    tasks = []
    # SBK is adding a QMin argument save_integral: saving RUNFILE, OrdInt and OneInt
    if 'pargrad' not in QMin and not QMin['save_integral']:
        tasks.append(['gateway'])
        if 'pcap' in QMin and 'ghost' in QMin:
            tasks.append(['ghost'])
        tasks.append(['seward'])

    # SBK added the following: The following snippet can be improved. #TODO
    if 'pcap' in QMin:
        mofile = ''
        if 'master' in QMin:
            if 'always_guess' not in QMin and 'scforb' not in QMin:
                if 'init' in QMin or 'always_orb_init' in QMin:
                    # Read initial RASSCF (not recommended) Guessorb from main directory `init` file.
                    ls = os.listdir(QMin['pwd'])
                    for i in ls:
                        if 'MOLCAS.JobIph.init' in i:
                            mofile = os.path.join(QMin['pwd'], 'MOLCAS.JobIph.init')
                            break
                        elif 'MOLCAS.RasOrb.init' in i:
                            mofile = os.path.join(QMin['pwd'], 'MOLCAS.RasOrb.init')
                            break
                elif 'samestep' in QMin:
                    mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')
                else:
                    mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph.old')  # This Can be changed SBK
                    # mofile = os.path.join(QMin['savedir'], 'MOLCAS.RasOrb.old')  # This is changed SBK (Nov 8, 24)

            # SBK added SCF guess option for "always_guess", not using the default SEWARD guess in MOLCAS
            elif 'always_guess' in QMin and 'scforb' not in QMin:
                # One possible choice but does not connect the consecutive time-steps' jobs.
                tasks.append(['scf'])
            # SBK; adding a new keyowrd 'scforb' for taking scf guess from previous step
            elif 'scforb' in QMin:
                # Consecutive time-steps are connected by ScfGuess (We are using this!)
                if 'init' in QMin:
                    ls = os.listdir(QMin['pwd'])
                    # print("This is Initial calculation.")
                    for i in ls:
                        if 'MOLCAS.ScfOrb.init' in i:
                            mofile = os.path.join(QMin['pwd'], 'MOLCAS.ScfOrb.init')
                            break
                    if mofile == '':
                        print("\nCan't find %s in :\t %s " % ('MOLCAS.ScfOrb.init', QMin['pwd']))
                        print(
                            "\nWarning: Please add the initial INPORB file or add 'always_guess' keyword for always SCF guess.\n"
                            "Continuing without it and performing SCF calculation for initial point\n")
                        # sys.exit(691)
                elif 'samestep' in QMin:
                    mofile = os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb')
                else:
                    mofile = os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb.old')
                # tasks.append(['link', mofile, 'INPORB'])
            #
            if not mofile == '':
                if 'JobIph' in mofile:
                    tasks.append(['link', mofile, 'JOBOLD'])
                elif 'RasOrb' in mofile:  # New file type ScfOrb
                    tasks.append(['link', mofile, 'INPORB'])
                elif 'ScfOrb' in mofile:
                    tasks.append(['link', mofile, 'INPORB'])  # Takes MOLCAS.ScfOrb.Old and uses for guess in SCF run
                    tasks.append(['scf'])  # New SCF orbitals are generated based on old ones and used in RASSCF
            if 'always_guess' not in QMin and mofile == '':  # otherwise two times &SCF is printed when 'always guess' is on.
                tasks.append(['scf'])

            if 'samestep' not in QMin or 'always_orb_init' in QMin:
                jobiph = 'JobIph' in mofile
                # rasorb = 'RasOrb' in  mofile  or "ScfOrb" in mofile
                # SBK: This above line is muted cause, we want rasorb==False, we calculate SCF
                # guess in new geometry based on old scf guess, task[4]==False, no LUMORB will be added.
                rasorb = 'RasOrb' in mofile
                tasks.append(['rasscf', 2, QMin['template']['roots'][0], jobiph, rasorb])  # SBK added SPIN=2 as 2nd arg
                # What about last two args?
                # They are false!
        # SBK Notes;
        # 'master': branch uses old jobiph file (last time step?)
        # 'master' completed JobIph/JOBOLD will be moved to savedir (curent time step) #TODO
        # 'branch_tranden', 'branch_nac', 'branch_gradient' needs task which copies this new JobIph
        #   from savediir to corresponding files (def: setupworkdir (.. ,WORKDIR))
        # Change the Jobiph format to JOBOLD later #TODO
        # Or add a link in MOLCAS.input file
        # rasorb is False for time-being #TODO

        if QMin['save_integral']:
            # SBK has made a keyword `DIRlink' which just copies file but does not print to MOLCAS.input
            OrdInt = os.path.join(QMin['savedir'], 'MOLCAS.OrdInt')
            OneInt = os.path.join(QMin['savedir'], 'MOLCAS.OneInt')
            RunFile = os.path.join(QMin['savedir'], 'MOLCAS.RunFile')
            tasks.append(['DIRlink', OrdInt, 'MOLCAS.OrdInt'])
            tasks.append(['DIRlink', OneInt, 'MOLCAS.OneInt'])
            tasks.append(['DIRlink', RunFile, 'MOLCAS.RunFile'])

        if QMin.get('branch_neutral'):
            mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')  # CAN create Rasorb if needed, not suggested
            tasks.append(['link', mofile, 'JOBOLD'])
            tasks.append(['rasscf', 1, 1, True, False])

        if QMin.get('branch_tranden'):
            mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')  # CAN create Rasorb if needed, not suggested
            tasks.append(['link', mofile, 'MOLCAS.JobIph'])
            tasks.append(['rassi', 'TRD1', QMin['template']['roots'][0]])
        if QMin.get('branch_nac'):
            mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')  # CAN create Rasorb if needed
            tasks.append(['link', mofile, 'MOLCAS.JobIph'])
            tasks.append(['alaska', 2, QMin['template']['roots'][0], True, False])
        if QMin.get('branch_gradient'):
            mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')  # CAN create Rasorb if needed
            tasks.append(['link', mofile, 'MOLCAS.JobIph'])
            tasks.append(['gradient', 2, QMin['template']['roots'][0], True, False])
        if QMin.get('track_roots') and 'adaptive_track' not in QMin:
            if 'track_wfoverlap' in QMin:
                mofile1 = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')
                tasks.append(['link', mofile1, 'MOLCAS.JobIph'])
            else:
                mofile1 = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')
                tasks.append(['link', mofile1, 'JOB001'])
                mofile2 = os.path.join(QMin['savedir'], 'MOLCAS.JobIph.old')
                tasks.append(['link', mofile2, 'JOB002'])
            tasks.append(['rassi', 'TRACK', QMin['template']['roots'][0]])
        elif QMin.get('track_roots') and 'adaptive_track' in QMin:
            _state, _ct = QMin['update_ref']
            mofile1 = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')
            tasks.append(['link', mofile1, 'JOB001'])
            mofile2 = os.path.join(QMin['savedir'], 'MOLCAS.JobIph.old.ref.%s' % (_ct + 1))
            tasks.append(['link', mofile2, 'JOB002'])
            tasks.append(['rassi', 'TRACK_%s' % _state, QMin['template']['roots'][0], _state])
        elif QMin.get('xms_corr') == True:
            mofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')  # CAN create Rasorb if needed
            tasks.append(['link', mofile, 'JOBOLD'])
            tasks.append(['rasscf', 2, QMin['template']['roots'][0], True, False])
            tasks.append(['caspt2', 'XMS', QMin['template']['roots'][0]])

    else:
        pass

    if DEBUG:
        printtasks(tasks)

    return tasks


# ======================================================================= #
def printGhostCoord(QMin, geomSUPPL=None, ghidx=1):
    """
    Print ghost atom in Centre-of-mass

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
        geo = QMin['geo']

    string = ""
    import isotope_mass
    mass_coord = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0
    for iatom, atom in enumerate(geo):
        total_mass += isotope_mass.MASSES.get(atom[0])
        _coord = np.array([isotope_mass.MASSES.get(atom[0]) * coord for coord in atom[1:]])
        mass_coord += _coord

    coord_com = (mass_coord / total_mass) * au2a
    string += "%s \t %12.10f \t %12.10f \t %12.10f \t %s" % (
        "  X%i" % ghidx, coord_com[0], coord_com[1], coord_com[2], QMin['unit'][0])
    return string


# Looks good

def writeMOLCASinput(tasks, QMin):
    # SBK: QMin will have specific information about branch of the calculation (TRD1, gradient, NAC)
    # SBK, this whole branch needs to be rewritten
    string = ''

    for task in tasks:

        if task[0] == 'gateway':
            string += '&GATEWAY\n Expert \n COORD=MOLCAS.xyz\n GROUP=NOSYM\n BASIS=%s\n' % (
                    QMin['template']['basis'])  # SBK added Expert Keyword

            if QMin['template']['baslib']:
                string += ' BASLIB\n %s\n' % QMin['template']['baslib']
            if QMin['template']['cholesky']:
                string += 'RICD\nCDTHreshold=%f\n' % (QMin['template']['cholesky_accu'])
            string += '\n'

        elif task[0] == 'ghost':
            try:
                with open(os.path.join(QMin['pwd'], 'X.basis'), 'r') as fbasis:
                    ghost_basis = fbasis.readlines()
                string += "\nExpert\n Basis set\n X ..... / inline\n"
                for line in ghost_basis:
                    if line.startswith('*'):
                        pass
                    else:
                        string += line
                string += "%s\n End of Basis set\n\n\n" % (printGhostCoord(QMin))
            except Exception as e:
                error_message = "Could not add any Ghost basis to the MOLCAS.input"
                detailed_traceback = traceback.format_exc()
                print(f"{error_message}\nError details:\n{detailed_traceback}")

        elif task[0] == 'seward':
            string += '&SEWARD\n'
        # SBK added a new keyword for guess instead of default molcas seward guess
        # We need it for PCAP calcaulation
        elif task[0] == 'scf':
            string += '\n&SCF\n'

        # SBK needs the following part for JOBIPH
        elif task[0] == 'link':
            name = os.path.basename(task[1])
            string += '>> COPY %s %s\n\n' % (name, task[2])
        elif task[0] == 'DIRlink':
            if 'integral_link' in QMin:
                string += '>> UNIX ln -fs %s %s\n\n' % (task[1], task[2])  # SBK doesn't recommend this!
            else:
                pass

        elif task[0] == 'copy':
            string += '>> COPY %s %s\n\n' % (task[1], task[2])

        elif task[0] == 'rm':
            string += '>> RM %s\n\n' % (task[1])

        elif task[0] == 'rasscf':
            nactel = QMin['template']['nactel']
            try:
                npad = QMin['template']['rootpad'][task[1] - 1]
            except:
                npad = 0
            if (nactel - task[1]) % 2 == 0:
                nactel -= 1
            string += '&RASSCF\n SPIN=%i\n NACTEL=%i 0 0\n INACTIVE=%i\n RAS2=%i\n ITERATIONS=%i,%i\n' % (
                task[1],
                nactel,
                QMin['template']['inactive'],
                QMin['template']['ras2'],
                QMin['template']['iterations'][0],
                QMin['template']['iterations'][1])

            # if 'master' not in QMin and 'branch_neutral' not in QMin:
            if 'master' not in QMin:
                string += " CIONLY\n CIRESTART\n"  # Jobiph in task is given true, JOBIPH will be added later.
            '''
            Note: if 'scforb' in QMin and 'master' in QMin: 
            #iFF: RasOrb/ScfOrb in task is given true, LUMORB will be added later.
             If False: LUMORB is not used, NO LUMORB in master branch. #UPDATE: 11/04
             SCF orbitals are used as guess in RASSCF.
                string += " LUMORB\n"'''

            #            string += ' CIROOT=%i %i 1\n' % (QMin['template']['roots'][0], QMin['template']['roots'][0])
            string += ' CIROOT=%i %i 1\n' % (task[2], task[2])
            string += ' ORBLISTING=NOTHING\n PRWF=0.1\n'

            if 'grad' in QMin and QMin['gradmode'] < 2:
                string += ' THRS=1.0e-10 1.0e-06 1.0e-06\n'
                #string += ' THRS=1.0e-8 1.0e-4 1.0e-4\n'
            else:
                string += ' THRS=%14.12f %14.12f %14.12f\n' % (
                    QMin['template']['rasscf_thrs_e'], QMin['template']['rasscf_thrs_rot'],
                    QMin['template']['rasscf_thrs_egrd'])
            if task[3]:
                string += ' JOBIPH\n'
            elif task[4]:
                string += ' LUMORB\n'
            if len(task) >= 6:
                for a in task[5]:
                    string += a + '\n'
            string += '\n'

        # SBK added
        elif task[0] == 'rassi':
            if QMin.get('branch_tranden'):
                string += '&RASSI\n TRD1\n'
                string += ' EJOB\n'
                # TODO Replace TRD1 by EJOB so that we don't have to recalculate HEFF since HEFF for CASSCF is diagonal
                # Read straight from the JOBMIX file (JOBIPH), Less expensive option for high number of states.
            # if task[1] == 'overlap':
            # string += ' OVERLAPS\n'
            elif 'master' in QMin:
                string += '&RASSI\n OVERLAPS\n'
            elif task[1] == 'TRACK':
                if 'track_all' in QMin and 'adaptive_track' not in QMin:
                    string += '&RASSI\n NROFJOBIPHS\n'
                    string += ' %i %i %i \n ' % (2, task[2], task[2])
                    # 2 JOBIPH files, nroots from first file JOB001, nroots from old file JOB002
                    string += ' '.join([str(j) for j in range(1, int(
                        task[2]) + 1)]) + '\n '  # DO we need all the roots from last time-step?
                    string += ' '.join([str(j) for j in range(1, int(task[2]) + 1)]) + '\n'
                    string += ' OVERLAPS\n'
                    string += ' EJOB\n'
                # TODO this is true for CASSCF and SS-CASPT2
                elif 'track_wfoverlap' in QMin:
                    string += '&RASSI\n '
                    string += ' CIPR\n THRS=0.000000d0\n'  # Only needs  CI coefficients read from JOBIPH.
                    string += ' EJOB \n '
                    # Can change the CIPR to 0.0000, print all coeffs? TODO#
                else:
                    string += '&RASSI\n NROFJOBIPHS\n'
                    string += ' %i %i %i \n ' % (2, task[2], 1)
                    # 2 JOBIPH files, nroots from first file JOB001, 1(neutral) from old file JOB002
                    string += ' '.join([str(j) for j in range(1, int(task[2]) + 1)]) + '\n'
                    string += ' %s\n' % (str(QMin['template']['neutral_state']))
                    string += ' OVERLAPS\n'
                    string += ' EJOB\n'
                # TODO this is true for CASSCF and SS-CASPT2

            elif 'TRACK_' in task[1]:
                string += '&RASSI\n NROFJOBIPHS\n'
                string += ' %i %i %i \n ' % (2, task[2], task[2])
                string += ' '.join([str(j) for j in range(1, int(task[2]) + 1)]) + '\n '
                string += ' '.join([str(j) for j in range(1, int(task[2]) + 1)]) + '\n '
                # string += " %s\n" % task[3]
                string += ' OVERLAPS\n'
                string += ' EJOB\n'
                # TODO this is true for CASSCF and SS-CASPT2

        elif task[0] == 'alaska':
            string += '&MCLR\n THRESHOLD=%s\n NAC = %s  %s \n\n' % (
                QMin['template']['gradaccudefault'], QMin['nac_pair'][0], QMin['nac_pair'][1])
            string += '&ALASKA\n'
            # SBK changed this incase there is convergence problem with run
        elif task[0] == 'gradient':
            string += '&MCLR\n THRESHOLD=%s\n SALA = %s\n\n' % (
                QMin['template']['gradaccudefault'], QMin['grad_root'][0])
            string += '&ALASKA\n'

        elif task[0] == 'caspt2':
            string += '&CASPT2\nSHIFT=0.0\nIMAGINARY=%5.3f\nIPEASHIFT=%4.2f\nMAXITER=%i\n' % (
                QMin['template']['imaginary'],
                QMin['template']['ipea'],
                200)
            if QMin['template']['frozen'] != -1:
                string += 'FROZEN=%i\n' % (QMin['template']['frozen'])
            if QMin['method'] == 1:
                string += 'NOMULT\n'
            if task[1] == 'XMS':
                string += 'XMULTISTATE= %i ' % (task[2])
            else:
                string += 'MULTISTATE= %i ' % (task[2])

            for i in range(task[2]):
                string += '%i ' % (i + 1)
            string += '\nOUTPUT=BRIEF\nPRWF=0.1\n'
            string += '\n'

    return string


# ======================================================================= #


def writegeomfile(QMin):
    string = ''
    string += '%i\n\n' % (QMin['natom'])
    for iatom, atom in enumerate(QMin['geo']):
        string += '%s%i ' % (atom[0], iatom + 1)
        for xyz in range(1, 4):
            string += ' %f' % (atom[xyz] * au2a)
        string += '\n'
    return string


# ======================================================================= #


def setupWORKDIR(WORKDIR, tasks, QMin):
    # print(WORKDIR, tasks)
    # mkdir the WORKDIR, or clean it if it exists, then copy all necessary JobIph files from pwd and savedir
    # then put the geom.xyz and MOLCAS.input files

    # set up the directory
    if os.path.exists(WORKDIR):
        if os.path.isfile(WORKDIR):
            print('%s exists and is a file!' % (WORKDIR))
            sys.exit(72)
        elif os.path.isdir(WORKDIR):
            if DEBUG:
                print('Remake\t%s' % WORKDIR)
            shutil.rmtree(WORKDIR)
            os.makedirs(WORKDIR)
    else:
        try:
            if DEBUG:
                print('Make\t%s' % WORKDIR)
            os.makedirs(WORKDIR)
        except OSError:
            print('Can not create %s\n' % (WORKDIR))
            sys.exit(73)

    # write geom file
    geomstring = writegeomfile(QMin)
    filename = os.path.join(WORKDIR, 'MOLCAS.xyz')
    writefile(filename, geomstring)
    if DEBUG:
        print(geomstring)
        print('Geom written to: %s' % (filename))

    # write MOLCAS.input
    inputstring = writeMOLCASinput(tasks, QMin)
    filename = os.path.join(WORKDIR, 'MOLCAS.input')
    writefile(filename, inputstring)
    if DEBUG:
        print(inputstring)
        print('MOLCAS input written to: %s' % (filename))

    # make subdirs
    if QMin['mpi_parallel']:
        for i in range(QMin['ncpu'] - 1):
            subdir = os.path.join(WORKDIR, 'tmp_%i' % (i + 1))
            os.makedirs(subdir)

    # JobIph copying
    copyfiles = set()
    for task in tasks:
        if task[0] == 'link' and task[1][0] == '/':
            copyfiles.add(task[1])
        # SBK added this copy command "DIRlink" which just copies file but does not print any shell cmd in MOLCAS.input
        if task[0] == 'DIRlink' and task[1][0] == '/':
            if 'integral_link' in QMin:  # This will link files by os.symlink, saves a lot of disk space!
                _tolink = os.path.join(WORKDIR, task[2])
                # link(task[1], _tolink) #Temporarily disabling it, using MOLCAS shell LINK EMIL
            else:
                copyfiles.add(task[1])
    #
    for files in copyfiles:
        if DEBUG:
            print('Copy:\t%s\n\t==>\n\t%s' % (files, WORKDIR))
        shutil.copy(files, WORKDIR)
        if QMin['mpi_parallel']:
            for i in range(QMin['ncpu'] - 1):
                subdir = os.path.join(WORKDIR, 'tmp_%i' % (i + 1))
                if DEBUG:
                    print('Copy:\t%s\n\t==>\n\t%s' % (files, subdir))
                shutil.copy(files, subdir)
    return


# ======================================================================= #
def runMOLCAS(WORKDIR, MOLCAS, driver, ncpu, strip=False):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    os.environ['WorkDir'] = WORKDIR
    os.environ['MOLCAS_NPROCS'] = str(ncpu)
    path = driver
    if not os.path.isfile(path):
        print('ERROR: could not find Molcas driver ("pymolcas" or "molcas.exe") in $MOLCAS/bin!')
        sys.exit(74)
    string = path + ' MOLCAS.input'
    stdoutfile = open(os.path.join(WORKDIR, 'MOLCAS.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'MOLCAS.err'), 'w')
    if PRINT or DEBUG:
        starttime = datetime.datetime.now()
        sys.stdout.write('START:\t%s\t%s\t"%s"\n' % (WORKDIR, starttime, string))
        sys.stdout.flush()
    try:
        runerror = sp.call(string, shell=True, stdout=stdoutfile, stderr=stderrfile)
        # pass
    except OSError as e:
        print('Call have had some serious problems:', e)
        sys.exit(75)
    stdoutfile.close()
    stderrfile.close()
    if PRINT or DEBUG:
        endtime = datetime.datetime.now()
        sys.stdout.write(
            'FINISH:\t%s\t%s\tRuntime: %s\tError Code: %i\n' % (WORKDIR, endtime, endtime - starttime, runerror))
        sys.stdout.flush()
    os.chdir(prevdir)

    if strip and not DEBUG:
        stripWORKDIR(WORKDIR)
    return runerror


# ======================================================================= #
def generate_joblist(QMin):
    '''split the full job into subtasks, each with a QMin dict, a WORKDIR
    structure: joblist = [ {WORKDIR: QMin, ..}, {..}, .. ]
    each element of the joblist is a set of jobs,
    and all jobs from the first set need to be completed before the second set can be processed.'''
    maxmem = int(QMin['ncpu_avail']) * int(QMin['memory'])
    joblist = []
    if QMin['gradmode'] == 0 or QMin['gradmode'] == 3:
        # case of serial gradients on one cpu
        QMin1 = deepcopy(QMin)
        QMin1['master'] = []
        if QMin['mpi_parallel']:
            QMin1['ncpu'] = QMin['ncpu']
        else:
            QMin1['ncpu'] = 1
        QMin['nslots_pool'] = [1]
        joblist.append({'master': QMin1})


    elif QMin['gradmode'] == 1:
        # case of analytical gradients for several states on several cpus

        # we will do wavefunction and dm, soc, overlap always first
        # afterwards we will do all gradients and nacdr asynchonously
        QMin1 = deepcopy(QMin)
        QMin1['master'] = []
        QMin1['keepintegrals'] = []
        QMin1['gradmap'] = []
        QMin1['nacmap'] = []
        if 'ion' in QMin:
            QMin1['keepintegrals'] = []
        if QMin['mpi_parallel']:
            QMin1['ncpu'] = QMin['ncpu']
        else:
            QMin1['ncpu'] = 1
        if 'pcap' in QMin:
            QMin1['ncpu'] = QMin['ncpu_avail']
            QMin1['memory'] = copy.deepcopy(maxmem)
        QMin['nslots_pool'] = [1]

        # SBK added this
        # QMin1['branch_tranden'] = False  # For TRD1 keyowrd in RASSI module, needs enormous computational time.
        # QMin1['branch_gradient'] = False  # Calculates all gradients asynchronously. May improve this later, but not sure.
        # QMin1['branch_nac'] = False
        # QMin1['track_roots'] = False
        # QMin1['save_integral'] = False

        joblist.append({'master': QMin1})  # main branch

        QMin2 = deepcopy(QMin)
        remove = ['h', 'soc', 'dm', 'always_guess', 'always_orb_init', 'comment', 'ncpu', 'init', 'veloc', 'overlap',
                  'ion']
        for r in remove:
            QMin2 = removekey(QMin2, r)

        QMin2 = deepcopy(QMin)  # NAC Branch
        QMin2['save_integral'] = True  # SBK added a new integral save routine

        QMin3 = deepcopy(QMin2)  # TRD1 Branch
        QMin4 = deepcopy(QMin2)  # Gradient branch
        QMin5 = deepcopy(QMin2)  # Neutral state tracking branch

        # QMin branch for neutral calculation
        QMin_neutral = copy.deepcopy(QMin)
        QMin_neutral['ncpu'] = deepcopy(QMin['ncpu'])
        QMin_neutral['branch_neutral'] = True
        QMin['nslots_pool'].append(1)

        joblist.append({'NEUTRAL': QMin_neutral})

        if 'xms_correction' in QMin:
            QMinxms = copy.deepcopy(QMin2)
            QMinxms['xms_corr'] = True
            joblist.append({'xms_corr': QMinxms})

        # SBK is relaying all branching information to all joblist:QMin (s)

        if QMin['mpi_parallel']:
            QMin2['ncpu'] = deepcopy(QMin['ncpu'])
        else:
            QMin2['ncpu'] = 1

        QMin3['branch_tranden'] = True
        QMin['nslots_pool'].append(1)

        if 'screening' in QMin['template']:
            QMin3['ncpu'] = QMin['ncpu_avail']  # Async job, exhaust all resources
            QMin3['memory'] *= QMin3['ncpu']
        joblist.append({"TRD1": QMin3})

        if 'newstep' in QMin or 'track_wfoverlap' in QMin:  # WfOverlap tracking needs TRACK folder in init as well!
            QMin5['track_roots'] = True
            # QMin5['ncpu'] = 1  # It is excruciatingly slow if RASSI overlaps are chosen as tracking: `track_all` keyword
            QMin['nslots_pool'].append(1)
            if 'track_wfoverlap' in QMin:
                QMin5['keepintegrals'] = []
            if 'adaptive_track' in QMin:
                # need multiple track jobs
                QMin5_neutral = deepcopy(QMin5)
                del QMin5_neutral['adaptive_track']
                del QMin5_neutral['track_all']
                joblist.append({'TRACK': QMin5_neutral})  # For Neutral

                for _ct, _state in enumerate(QMin['template']['act_states']):
                    QMin5['update_ref'] = [_state, _ct]
                    joblist.append({'TRACK_%s' % _state: QMin5})
            else:
                joblist.append({'TRACK': QMin5})  # Only one track job

        if 'grad' in QMin and 'pcap' in QMin:
            for istate1 in range(QMin['template']['roots'][0]):
                QMin4['branch_gradient'] = True
                QMin4['grad_root'] = [istate1 + 1]
                _jobname = "grad_%s" % (istate1 + 1)
                joblist.append({str(_jobname): QMin4})
                QMin4 = removekey(QMin4, 'grad_root')
                QMin['nslots_pool'].append(1)
                for istate2 in range(istate1 + 1, QMin['template']['roots'][0]):
                    QMin2['branch_nac'] = True
                    QMin2['nac_pair'] = [istate1 + 1, istate2 + 1]
                    _jobname = "nac_%s_%s" % (istate1 + 1, istate2 + 1)
                    joblist.append({str(_jobname): QMin2})
                    QMin2 = removekey(QMin2, 'nac_pair')  # This line is necessary, Learned it hard way!!
                    QMin['nslots_pool'].append(1)

    if DEBUG:
        pprint.pprint(joblist, depth=3)
    return QMin, joblist


# ======================================================================= #
def run_calc_async(WORKDIR, QMin):
    err = 96
    irun = -1
    while err == 96:
        irun += 1
        if irun >= 2:
            print('CRASHED:\t%s\tDid not converge after %i tries.' % (WORKDIR, irun))
            return 96, irun
        try:
            Tasks = gettasks(QMin)
            if irun > 0:
                saveJobIphs(WORKDIR, QMin)
                if 'scforb' in QMin:
                    fromfileSCFORB = os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb')
                    tofileSCFORB = os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb.old')
                elif 'rasorb' in QMin:
                    fromfileSCFORB = os.path.join(QMin['savedir'], 'MOLCAS.RasOrb')
                    tofileSCFORB = os.path.join(QMin['savedir'], 'MOLCAS.RasOrb.old')

                try:
                    shutil.copy(fromfileSCFORB, tofileSCFORB)
                except FileNotFoundError:
                    print(f"The file {fromfileSCFORB} does not exist.")
                except Exception as e:
                    print(f"An error occurred while copying the file: {e}")

                # If convergence fails in master branch,
                # start with the most recent Scf orbs as guess from irun-1 run.
            setupWORKDIR(WORKDIR, Tasks, QMin)
            strip = 'keepintegrals' not in QMin
            err = runMOLCAS(WORKDIR, QMin['molcas'], QMin['driver'], QMin['ncpu'], strip)
        except Exception as problem:
            print('*' * 50 + '\nException in run_calc_async(%s)!' % (WORKDIR))
            traceback.print_exc()
            print('*' * 50 + '\n')
            raise problem
    return err, irun
    # Returning irun as well to find how many runs for a particular joblist.


def run_calc(WORKDIR, QMin):
    err = 96
    irun = -1
    while err == 96:
        irun += 1
        if 'grad' in QMin:
            if irun > 0:
                QMin['template']['gradaccudefault'] *= 10.
            if QMin['template']['gradaccudefault'] > QMin['template']['gradaccumax']:
                print('CRASHED:\t%s\tMCLR did not converge.' % (WORKDIR))
                return 96, irun
        else:
            if irun > 0:
                print('CRASHED:\t%s\tDid not converge.' % (WORKDIR))
                return 96, irun
        if irun >= 10:
            print('CRASHED:\t%s\tDid not converge. Giving up after 10 tries.' % (WORKDIR))
        try:
            Tasks = gettasks(QMin)
            setupWORKDIR(WORKDIR, Tasks, QMin)
            strip = 'keepintegrals' not in QMin
            err = runMOLCAS(WORKDIR, QMin['molcas'], QMin['driver'], QMin['ncpu'], strip)
        except Exception as problem:
            print('*' * 50 + '\nException in run_calc(%s)!' % (WORKDIR))
            traceback.print_exc()
            print('*' * 50 + '\n')
            raise problem
    # print(err, irun)
    return err, irun  # Returning irun as well to find how many runs for a particular joblist.


def screening_joblist(QMin, WORKDIR):
    '''
    Identifies the NAC jobs that will have minimum contribution.
    The screening parameter threshold will determine the screening degree.
    `screening  float(value)` line should be given in MOLCAS.template

    Parameters:
    ----------
    dict: QMin
    str: WORKDIR

    Returns:
    -------
     dictionary of jobs to be removed,
        Number of jobs to be skipped.
    '''
    outfilename = os.path.join(WORKDIR, 'MOLCAS.out')
    nroots = QMin['template']['roots'][0]
    screening_param = QMin['template']['screening']

    H0, W, _, _ = parse_MOLCAS_Hamiltonian(outfilename, QMin)
    H_CAP = H0 + float(QMin['template']['cap_eta']) * (1E-5) * W
    Leigvc, Reigvc, eVs = diagonalize_cmatrix(H_CAP)

    ct = 0
    removed_jobs = []
    for i in range(nroots):
        for j in range(i + 1, nroots):
            if abs(Leigvc[i, j] * Reigvc[i, j]) < screening_param and abs(
                    Leigvc[j, i] * Reigvc[j, i]) < screening_param:
                ct += 1
                removed_jobs.append("nac_%s_%s" % (i + 1, j + 1))

    RotMATfile = os.path.join(WORKDIR, 'SAVE.HCAP.h5')
    with h5py.File(RotMATfile, "w") as f:
        f.create_dataset("H0", data=H0)
        f.create_dataset("H_CAP", data=W)
    if DEBUG:
        print("\nScreening calculated and saved Rotation matrix and vectors in %s.\n\n" % WORKDIR)
    return removed_jobs, ct


def _geomfromMOLCASxyz(xyzfilename):
    '''
    Reads .xyz file in MOLCAS format and creates coordinate dictionary
    '''
    lines = readfile(xyzfilename)
    coords = []
    for _, line in enumerate(lines):
        if len(line) > 3:
            columns = line.split()
            label, x, y, z = columns
            if label.startswith('X') == False:
                coords.append((label[:-1], float(x), float(y), float(z)))
    return coords


def _doublemolMOLCAS(WORKDIR, QMin, kabsch=False, guessorb=False):
    '''
    Creates MOLCAS.xyz and MOLCAS.input for doublemol calculation in MOLCAS
    :param WORKDIR: string, Where to run the job
    :param QMin: dictionary,  containing all QM run info
    :param kabsch: bool, if True, kabsch rotation and translation is performed on new geometry
    :param guessorb: bool, if True, create guessorb.h5 file to extract the mixed ao-overlap
    :return: A directory set-up for DOUBLEMOL calculation
    '''
    _MOLCAS_dRmin = 0.000001

    def _xyz4pyscf(lines):
        atom_count = int(QMin['natom']) + 1 if 'ghost' in QMin else int(QMin['natom'])
        coords = []
        for _idx, line in enumerate(lines):
            if "Cartesian Coordinates" in line:
                for _xyzline in (lines[_idx + 4:_idx + 4 + atom_count]):
                    columns = _xyzline.split()
                    label, x, y, z = columns[1:5]
                    coords.append((label, float(x) * au2a, float(y) * au2a, float(z) * au2a))
                break
        return coords

    def remove_numeric_suffix(array):
        new_array = []
        for item in array:
            label = ''.join([i for i in item[0] if not i.isdigit()])
            new_item = (label,) + item[1:]
            new_array.append(new_item)
        return new_array

    def short_distance_check(geoold, geonew):
        '''
        Checks for short distance error between Atoms and modifies the NEW bond lengths tuple slightly
        '''
        molcas_short_dist_FIX = False
        while not molcas_short_dist_FIX:
            molcas_short_dist_FIX = True
            for idx, (dOLD, dNEW) in enumerate(zip(geoold, geonew)):
                _dr = np.sqrt(sum([(i - j) ** 2.0 for (i, j) in zip(dOLD[1:], dNEW[1:])]))

                if np.round(_dr, 6) <= _MOLCAS_dRmin:
                    print(
                        "\n\nWarning: Short distance encountered in DOUBLEMOL setup \n(Bond Length = %s, %s: %10.6f Angstrom): changing coordinates slightly \n" % (
                        dOLD[0], dNEW[0], _dr))
                    print("Before Modifying:\t %s %10.6f, %10.6f, %10.6f" % (dNEW[0], dNEW[1], dNEW[2], dNEW[3]))
                    geonew[idx] = list(geonew[idx])
                    _sign = 1.0 if geonew[idx][-1] == 0.0 else np.sign(geonew[idx][-1])
                    geonew[idx][-1] += _sign * _MOLCAS_dRmin * 10.0
                    geonew[idx] = tuple(geonew[idx])
                    print("After Modifying:\t %s %10.6f, %10.6f, %10.6f\n\n" % (
                    geonew[idx][0], geonew[idx][1], geonew[idx][2], geonew[idx][3]))
                    molcas_short_dist_FIX = False
        return geoold, geonew

    # Extract geometries from the MOLCAS.out in master/branch to get correct ordering of the atoms!
    oldfile = os.path.join(QMin['savedir'], "MOLCAS.geominfo.old")
    newfile = os.path.join(QMin['scratchdir'], 'master', "MOLCAS.out")
    geooldlines = readfile(oldfile)
    geonewlines = readfile(newfile)

    geoold = remove_numeric_suffix(_xyz4pyscf(geooldlines))
    geonew = remove_numeric_suffix(_xyz4pyscf(geonewlines))
    geoold, geonew = short_distance_check(geoold, geonew)
    '''
    oldfile = os.path.join(QMin['savedir'], "MOLCAS.xyz.old")
    newfile = os.path.join(QMin['scratchdir'], 'master', "MOLCAS.xyz")
    geoold = _geomfromMOLCASxyz(oldfile)
    geonew = _geomfromMOLCASxyz(newfile)
    '''

    if kabsch:
        geonew = align_geometry_Kabsch(geoold, geonew, QMin)

    doublegeomstring = ''
    doublegeomstring += '%i\n\n' % (int(len(geoold) + len(geonew)))
    iatom = 1
    for _geo in [geoold, geonew]:
        for _, atom in enumerate(_geo):
            doublegeomstring += '%s%i ' % (atom[0], iatom)
            iatom += 1
            for xyz in range(1, 4):
                doublegeomstring += ' %f' % (atom[xyz])
            doublegeomstring += '\n'

    # write doubleMOLCAS.xyz
    filename = os.path.join(WORKDIR, 'doubleMOLCAS.xyz')
    writefile(filename, doublegeomstring)
    if DEBUG:
        print('\nGeom written to: %s' % (filename))

    string = ''
    string += '&GATEWAY\n Expert \n COORD=doubleMOLCAS.xyz\n GROUP=NOSYM\n BASIS=%s\n' % (QMin['template']['basis'])
    if QMin['template']['baslib']:
        string += ' BASLIB\n%s\n' % QMin['template']['baslib']
    else:
        BASlib_molcas = os.path.join(QMin['molcas'], 'basis_library', str(QMin['template']['basis'].upper()))
        shutil.copy(BASlib_molcas, QMin['pwd'])
        if DEBUG:
            print('Copy:\t%s\n\t==>\n\t%s' % (BASlib_molcas, QMin['pwd']))

        string += ' BASLIB\n%s\n\n\n' % QMin['pwd']

    if guessorb:
        string += '''&SEWARD\n ONEOnly\n EXPErt\n NODElete\n NODKroll\n MULTipoles\n 0\n\n\n\n'''
    else:
        string += '''&SEWARD\n ONEOnly\n EXPErt\n NOGUessorb\n NODElete\n NODKroll\n MULTipoles\n 0\n\n\n\n'''

    # write double MOLCAS.input
    filename = os.path.join(WORKDIR, 'MOLCAS.input')
    writefile(filename, string)
    if DEBUG:
        print('MOLCAS input written to: %s' % (filename))

    # Print the ANO-RCC for ghost basis
    if QMin.get('ghost'):
        try:
            if QMin['template']['baslib']:
                ghost_ANORCC_BASFILE = os.path.join(QMin['template']['baslib'].rstrip('\n/'), 'ANO-RCC')
            else:
                ghost_ANORCC_BASFILE = os.path.join(QMin['pwd'].rstrip('\n/'), 'ANO-RCC')

            with open(os.path.join(QMin['pwd'], 'X.basis'), 'r') as fbasis:
                ghost_basis_lines = fbasis.readlines()

            BASstring = ''
            BASstring += '\n#Hamiltonian RH_\n#Nucleus PN_\n#Contraction ANO\n#AllElectron AE_\n\n\n'
            hstring = "".join(i for i in QMin['ghost'].split('.'))
            hstring2 = ", ".join(i for i in QMin['ghost'].split('.'))
            BASstring = "/X.ANO-RCC...%s.%s.\nDiffuse Ghost\nGHOST (%s) -> [%s]\n" % (
            hstring, hstring, hstring2, hstring2)

            for line in ghost_basis_lines:
                if line.startswith('*'):
                    pass
                else:
                    BASstring += line

        except Exception as e:
            error_message = "Could not add any Ghost basis to the MOLCAS.input"
            detailed_traceback = traceback.format_exc()
            print(f"{error_message}\nError details:\n{detailed_traceback}")

        writefile(ghost_ANORCC_BASFILE, BASstring)

    return


def setupdoublemolDIR(WORKDIR, QMin, kabsch, guessorb):
    """
    Creates DOUBLEMOL directory inside the TRACK directory of WFOVERLAP run.
    Contains MOLCAS.input and doubleMOLCAS.xyz to run doublemol calculation

    Parameters:
    ----------
    WORKDIR
    Working directory of WF-OVERLAP

    QMin
    dict: QMin dictionary

    kabsch
    bool: if kabsch alignment is needed or not

    guessorb:
    bool: if doubleMOLCAS.guessorb.h5 is accessed to read mixed overlap

    Returns:
    -------
    A directory set to run doublemol calculation

    """
    if os.path.exists(WORKDIR):
        if os.path.isfile(WORKDIR):
            print('%s exists and is a file!' % (WORKDIR))
            sys.exit(72)
        elif os.path.isdir(WORKDIR):
            if DEBUG:
                print('Remake\t%s' % WORKDIR)
            shutil.rmtree(WORKDIR)
            os.makedirs(WORKDIR)
    else:
        try:
            if DEBUG:
                print('Make\t%s' % WORKDIR)
            os.makedirs(WORKDIR)
        except OSError:
            print('Can not create %s\n' % (WORKDIR))
            sys.exit(73)
    _doublemolMOLCAS(WORKDIR, QMin, kabsch, guessorb)
    return


def run_double_mol(WORKDIR, QMin, kabsch, guessorb=False):
    """
    Runs doublemol molcas job.

    :param WORKDIR: string, Where to run the job
    :param QMin: dictionary,  containing all QM run info
    :param kabsch: bool, if True, kabsch rotation and translation is performed on new geometry
    :param guessorb: bool, if True, guessorb is stored in HDF5 file
    :return: A directory set-up for DOUBLEMOL calculation
    """
    err = 96
    irun = -1
    while err == 96:
        irun += 1
        try:
            setupdoublemolDIR(WORKDIR, QMin, kabsch, guessorb)
            strip = 'keepintegrals' not in QMin
            err = runMOLCAS(WORKDIR, QMin['molcas'], QMin['driver'], QMin['ncpu'], strip)
        except Exception as problem:
            print('*' * 50 + '\nException in run_calc(%s)!' % (WORKDIR))
            traceback.print_exc()
            print('*' * 50 + '\n')
            raise problem
    return err, irun


def setupWfOverlapDIR(WORKDIR, QMin, kabsch=False):
    '''
    Sets up working directory, prepare files for wave function overlap wfoverlap.x

    param: WORKDIR, absolute path of WORKDIR where WfOverlap should run.
    dict: QMin dictionary holding all information.
    bool: kabsch, whether Kabsch algorithm is invoked
    Returns:
        A working directory with all necessary files for wfoverlap.x run
    '''
    # RasOrb files
    oldRasOrbFROM = os.path.join(QMin['savedir'], 'MOLCAS.RasOrb.old')
    newRasOrbFROM = os.path.join(QMin['savedir'], 'MOLCAS.RasOrb')
    oldRasOrbTO = os.path.join(WORKDIR, 'MOLCAS.RasOrb.old')
    newRasOrbTO = os.path.join(WORKDIR, 'MOLCAS.RasOrb')

    link(oldRasOrbFROM, oldRasOrbTO)
    link(newRasOrbFROM, newRasOrbTO)
    # SBK can't assume that S^AO does not change, hence INSTEAD of RUNFILE and ONEINT from current iteration,
    # SBK now doing double mol calculation in MOLCAS
    _DMOLWORKDIR = os.path.join(QMin['scratchdir'], 'DOUBLEMOL')
    errorcode, _ = run_double_mol(_DMOLWORKDIR, QMin, kabsch)

    link(os.path.join(_DMOLWORKDIR, "MOLCAS.RunFile"), os.path.join(WORKDIR, "RUNFILE"))
    link(os.path.join(_DMOLWORKDIR, "MOLCAS.OneInt"), os.path.join(WORKDIR, "ONEINT"))

    if errorcode == 0:
        link(os.path.join(_DMOLWORKDIR, "MOLCAS.RunFile"), os.path.join(WORKDIR, "RUNFILE"))
        link(os.path.join(_DMOLWORKDIR, "MOLCAS.OneInt"), os.path.join(WORKDIR, "ONEINT"))
        same_aos = ''
    else:
        print("\n\n===> Warning: Double mol calculation returned with error, Taking AO overlap from current geometry."
              "\n This is a big approximation!!")
        link(os.path.join(QMin['savedir'], "MOLCAS.RunFile"), os.path.join(WORKDIR, "RUNFILE"))
        link(os.path.join(QMin['savedir'], "MOLCAS.OneInt"), os.path.join(WORKDIR, "ONEINT"))
        same_aos = 'same_aos'

    string = '''
    ao_read=1
    a_mo=MOLCAS.RasOrb.old
    b_mo=MOLCAS.RasOrb
    a_det=dets.old
    b_det=dets.new
    a_mo_read=1
    b_mo_read=1
    %s
    ''' % same_aos
    # Prepare input file dyson.in , funny it is that this is not a dyson calculation.
    inputfile = os.path.join(WORKDIR, 'dyson.in')
    writefile(inputfile, string)

    # Molcas env and other files like ONEINT and RunFile
    # TODO: may have to fix numbers of file copying
    # Generate dets.old and dets.new
    _fileOLD = os.path.join(QMin['savedir'], "MOLCAS.out.old")
    _outOLD = readfile(_fileOLD)
    _pathOLD = os.path.join(WORKDIR, "dets.old")
    ci_vectorsOLD = get_determinants(_outOLD, 2)
    string_CIVEC = format_ci_vectors(ci_vectorsOLD)
    writefile(_pathOLD, string_CIVEC)
    # format_ci_diag_vectors()

    _pathNEW = os.path.join(WORKDIR, 'MOLCAS.out')
    _outNEW = readfile(_pathNEW)
    _pathNEW = os.path.join(WORKDIR, "dets.new")
    ci_vectorsNEW = get_determinants(_outNEW, 2)
    string_CIVEC = format_ci_vectors(ci_vectorsNEW)
    writefile(_pathNEW, string_CIVEC)

    return


# ======================================================================= #


def runjobs(joblist, QMin):
    # Notes to SBK:
    # needs to resturcutue the workflow :
    # 'master' branch completes the job
    # JobIphs are copied and moved to savedir
    # Other branches take that jobfile and run in sync
    # TODO Can create an option that if the 'master' job fails, copy JobIph.old
    # which is converged.

    # SBK added a runcount variable.

    if 'newstep' in QMin:
        moveJobIphs(QMin)

    print('>>>>>>>>>>>>> Starting the job execution')

    errorcodes = {}
    runcount = {}
    # SBK added here
    if 'pcap' in QMin:
        for ijobset, jobset in enumerate(joblist):
            if not jobset:
                continue
            else:
                for job in jobset:
                    if job != 'master':
                        continue
                    else:
                        # job=='master', SBK made this complicated
                        print("\n\n\tThis is master branch: Projected-CAP calculation, \n\tGenerating JobIph...\n\n")
                        QMin1 = jobset[job]
                        WORKDIR = os.path.join(QMin['scratchdir'], job)
                        errorcodes[job], runcount[job] = run_calc_async(WORKDIR, QMin1)  # SBK: Async Job
                        # WORKDIR = os.path.join(QMin['scratchdir'], 'master')
                        saveJobIphs(WORKDIR, jobset[job])
                        saveIntRun(WORKDIR, jobset[job])
                        joblist = [d for d in joblist if job not in d]  # SBK: removing the 'master' branch.
                        if errorcodes[job] != 0:
                            print("Orbital generation in master branch failed.")
                            sys.exit(751)
                        time.sleep(QMin['delay'])
                        # 'master' branch has 'keepintegrals'=True, All out puts will be saved by default!

                        # Screening can be implemented very easily! Only NAC jobs will be screened cause they are plenty!
                        if 'screening' in QMin['template']:
                            async_job = "TRD1"

                            print("===> Screening of NAC jobs are ON"
                                  "\n\n\tRunning in TRD1 branch: Screening via L^T.R matrix elements, \n\tGenerating rassi.h5...\n\n")

                            QMin2 = [d[async_job] for d in joblist if async_job in d][0]
                            WORKDIR2 = os.path.join(QMin['scratchdir'], async_job)
                            errorcodes[async_job], runcount[async_job] = run_calc_async(WORKDIR2, QMin2)
                            joblist = [d for d in joblist if
                                       async_job not in d]  # Remove the job, no need to calculate again!!
                            if errorcodes[job] != 0:
                                print("Error: rassi.h5 generation in TRD1 branch failed. Screening cannot work!")
                                sys.exit(752)

                            removed_jobs, _ct = screening_joblist(QMin, WORKDIR2)
                            print("\n\t%s jobs will be screened out. These are the following:\n\n" % _ct)
                            print('\n'.join(['\t'.join(job.ljust(max(map(len, removed_jobs)) + 3)
                                                       for job in removed_jobs[i:i + 10])
                                             for i in range(0, len(removed_jobs), 10)]), '\n\n')

                            for _, sadjobs in enumerate(removed_jobs):
                                # print("\t\t Job: %s skipped! :( " % sadjobs)
                                joblist = [d for d in joblist if sadjobs not in d]
                            time.sleep(QMin['delay'])
                        '''
                        if 'track_wfoverlap' in QMin or 'track_natorb' in QMin:
                            shutil.copy(os.path.join(WORKDIR, "MOLCAS.out"),
                                        os.path.join(QMin['savedir'], "MOLCAS.geominfo.old"))
                        '''

    #
    _wfoverlap_job = []
    _runcalc_args = []
    for ijobset, jobset in enumerate(joblist):
        if not jobset:
            continue
        for job in jobset:
            QMin1 = jobset[job]
            WORKDIR = os.path.join(QMin['scratchdir'], job)
            _runcalc_args.append((WORKDIR, QMin1))
            if job == 'TRACK' and 'track_wfoverlap' in QMin:
                _wfoverlap_job.append((WORKDIR, QMin1))
                # _runcalc_args = _runcalc_args[:-1]

    # SBK changed the following subroutine
    if 'pcap' in QMin:
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            results = executor.map(run_calc, *zip(*_runcalc_args))
            for i, jobset in enumerate(joblist):
                if not jobset:
                    continue
                for job in jobset:
                    errorcodes[job], runcount[job] = next(results)
                    time.sleep(QMin['delay'])

        for _, (WORKDIR, QMin1) in enumerate(_wfoverlap_job):
            if 'newstep' in QMin:
                rerunWFOVLP = False
                ct = 1
                while True:
                    setupWfOverlapDIR(WORKDIR, QMin1, rerunWFOVLP)
                    runerror = runWFOVERLAPS(WORKDIR, QMin1['wfoverlap'], QMin1['memory'], QMin1['ncpu_avail'])
                    errorcodes['TRACK'], runcount['TRACK'] = runerror, ct
                    ct += 1
                    if runerror == 0:
                        rerunWFOVLP = need_alignment(WORKDIR, QMin)
                    if rerunWFOVLP == False:
                        break
            files_TO_copy = [os.path.join(WORKDIR, "MOLCAS.out"),
                             os.path.join(WORKDIR, "MOLCAS.xyz")]
            for file in files_TO_copy:
                shutil.copy(file, os.path.join(QMin['savedir'], f'{os.path.basename(file)}.old'))
            geomnewfile = os.path.join(QMin['scratchdir'], 'master', "MOLCAS.out")
            shutil.copy(geomnewfile, os.path.join(QMin['savedir'], "MOLCAS.geominfo.old"))

    else:
        for ijobset, jobset in enumerate(joblist):
            if not jobset:
                continue
            pool = Pool(processes=QMin['nslots_pool'][ijobset])
            for job in jobset:
                QMin1 = jobset[job]
                WORKDIR = os.path.join(QMin['scratchdir'], job)
                errorcodes[job], runcount[job] = pool.apply_async(run_calc, [WORKDIR, QMin1])  # SBK CHANGED HERE
                time.sleep(QMin['delay'])
            pool.close()
            pool.join()

            if 'master' in jobset and 'pcap' not in QMin:  # SBk added
                WORKDIR = os.path.join(QMin['scratchdir'], 'master')
                saveJobIphs(WORKDIR, jobset['master'])

            print('')

    for i in errorcodes:
        try:
            errorcodes[i] = errorcodes[i].get()
        except:
            errorcodes[i] = errorcodes[i]
            # SBK added this for 'master' branch, cause it runs serially
            # and does not return an instance by run_calc unlike the pool.map

    for i in runcount:
        try:
            runcount[i] = runcount[i].get()
        except:
            runcount[i] = runcount[i]

    if PRINT:
        string = '  ' + '=' * 40 + '\n'
        string += '||' + ' ' * 40 + '||\n'
        string += '||' + ' ' * 10 + 'All Tasks completed!' + ' ' * 10 + '||\n'
        string += '||' + ' ' * 40 + '||\n'
        string += '  ' + '=' * 40 + '\n'
        print(string)
        j = 0
        string = 'Error Codes (Run count):\n\n'
        for i in errorcodes:
            string += '\t%s\t%i (%i)' % (i + ' ' * (10 - len(i)), errorcodes[i], (runcount[i] + 1))
            j += 1
            if j == 4:
                j = 0
                string += '\n'
        print(string)

    if any((i != 0 for i in errorcodes.values())):
        print('Some subprocesses did not finish successfully!')
        # sys.exit(76)

    return errorcodes


# ======================================================================= #
def message_string(message):
    _len_message = int(len(message) / 2)
    _message_print = '\n \n'
    _message_print += "*" * int(2 * _len_message + 10)
    _message_print += '\n'
    _message_print += "%s %s %s" % ("*" * 1, message, "*" * 1)
    _message_print += '\n'
    _message_print += "*" * int(2 * _len_message + 10)
    _message_print += '\n \n'
    return _message_print


def CollectallNACjobs(joblist, QMin, errorcodes):
    _NACoutfilepath = os.path.join(QMin['scratchdir'], 'master', "MOLCAS_GRADNAC")
    _NACoutfile = open(_NACoutfilepath, 'w')

    for jobset in joblist:
        for job in jobset:
            if job == 'master' or job == 'TRD1':
                pass
            else:
                # if errorcodes[job] == 0:
                if errorcodes.get(job, -1) == 0:
                    # A little precaution: when screening on, it removes certain jobs!
                    _fname = 'MOLCAS.out'
                    outfile = os.path.join(QMin['scratchdir'], job, _fname)
                    if DEBUG:
                        print('Merging %s ---to--- %s' % (outfile, _NACoutfilepath))
                    with open(outfile, 'r') as _readoutfile:
                        outfile_contents = _readoutfile.read()
                    if job == 'NEUTRAL':
                        _NACoutfile.write(message_string(f"Merging {job.upper()} jobfile %s " % outfile))
                    else:
                        _NACoutfile.write(message_string("Merging %s " % outfile))
                    _NACoutfile.write(outfile_contents)

    _NACoutfile.close()
    NACoutfile = readfile(_NACoutfilepath)
    return _NACoutfilepath, NACoutfile


# ======================================================================= #
def collectOutputsPCAP(joblist, QMin, errorcodes):
    QMout = {}
    tdmoutfilename = os.path.join(QMin['scratchdir'], 'TRD1', 'MOLCAS.out')
    xmscaspt2filename = os.path.join(QMin['scratchdir'], 'xms_corr', 'MOLCAS.out')
    neutral_filename = os.path.join(QMin['scratchdir'], 'NEUTRAL', 'MOLCAS.out')

    if 'grad' in QMin and 'pcap' in QMin:
        GRADNACoutfilePath, GRADNACoutfile = CollectallNACjobs(joblist, QMin, errorcodes)
        print('>> Reading %s' % (GRADNACoutfilePath))
        QMout['master'] = getQMout(GRADNACoutfile, QMin, outfilename=tdmoutfilename)
    else:
        QMout['master'] = getQMout(readfile(tdmoutfilename), QMin, outfilename=tdmoutfilename)

    return QMout


def readXMStransform(QMin, fXMS):
    flines = readfile(fXMS)
    nroots = QMin['template']['roots'][0]
    U_xms = np.zeros([nroots, nroots])

    colmax = 5 if nroots > 5 else nroots
    colmin = nroots % 5 if nroots % 5 != 0 else 5
    cols = []
    _SEG_CT = 0
    irow = 0

    stopstring = "***********************************************"
    for idx, line in enumerate(flines):
        if 'H0 eigenvectors:' in line:
            for linex in flines[idx + 1:]:
                if stopstring in linex:
                    break
                content = linex.split()
                if len(content) == 0:
                    continue

                if len(content) == colmax:
                    _SEG_CT += 1
                    cols = [int(i) - 1 for i in content]

                if len(content) == colmax + 1:
                    irow = int(content[0]) - 1
                    for idx, icol in enumerate(cols):
                        U_xms[irow, icol] = content[1:][idx]

                if _SEG_CT == int(nroots / 5) and irow + 1 == nroots:
                    colmax = colmin
            break

    return U_xms


# ======================================================================= #

def loewdin_orthonormalization(A):
    '''
    returns loewdin orthonormalized matrix
    '''

    # S = A^T * A
    S = np.dot(A.T, A)

    # S^d = U^T * S * U
    S_diag_only, U = np.linalg.eigh(S)

    # calculate the inverse sqrt of the diagonal matrix
    S_diag_only_inverse_sqrt = [1. / (float(d) ** 0.5) for d in S_diag_only]
    S_diag_inverse_sqrt = np.diag(S_diag_only_inverse_sqrt)

    # calculate inverse sqrt of S
    S_inverse_sqrt = np.dot(np.dot(U, S_diag_inverse_sqrt), U.T)

    # calculate loewdin orthonormalized matrix
    A_lo = np.dot(A, S_inverse_sqrt)

    # normalize A_lo
    A_lo = A_lo.T
    length = len(A_lo)
    A_lon = np.zeros((length, length), dtype=complex)

    for i in range(length):
        norm_of_col = np.linalg.norm(A_lo[i])
        A_lon[i] = [e / (norm_of_col ** 0.5) for e in A_lo[i]][0]

    return A_lon.T


# ======================================================================= #


def arrangeQMout(QMin, QMoutall, QMoutDyson):
    QMout = {}
    if 'h' in QMin or 'soc' in QMin:
        if 'pcap' in QMin:
            QMout['h'] = QMoutall['master']['h_diag_res_real']
        else:
            QMout['h'] = QMoutall['master']['h']
    if 'dm' in QMin:
        QMout['dm'] = QMoutall['master']['dm']
    if 'overlap' in QMin:
        QMout['overlap'] = QMoutall['master']['overlap']
    # Phases from overlaps
    if 'phases' in QMin:
        if 'phases' not in QMout:
            QMout['phases'] = [complex(1., 0.) for i in range(QMin['nmstates'])]
        if 'overlap' in QMout:
            for i in range(QMin['nmstates']):
                if QMout['overlap'][i][i].real < 0.:
                    QMout['phases'][i] = complex(-1., 0.)

    if 'grad' in QMin and not 'pcap' in QMin:
        if QMin['gradmode'] == 0 or QMin['gradmode'] == 3:
            QMout['grad'] = QMoutall['master']['grad']

        elif QMin['gradmode'] == 1:
            zerograd = [[0.0 for xyz in range(3)] for iatom in range(QMin['natom'])]
            grad = []
            for i in sorted(QMin['statemap']):
                mult, state, ms = tuple(QMin['statemap'][i])
                if (mult, state) in QMin['gradmap']:
                    name = 'grad_%i_%i' % (mult, state)
                    grad.append(QMoutall[name]['grad'][i - 1])
                else:
                    grad.append(zerograd)
            QMout['grad'] = grad

    if 'grad' in QMin and 'pcap' in QMin:
        QMout['grad'] = QMoutall['master']['grad_diag_res']
    if 'nacdr' in QMin and 'pcap' in QMin:
        QMout['nacdr'] = QMoutall['master']['nacdr_diag_res']
    if PRINT:
        print('\n===================================================================')
        print('========================= Final Results ===========================')
        print('===================================================================')
        printQMout(QMin, QMout)

    return QMout


# ======================================================================= #


def getcaspt2weight(out, mult, state):
    modulestring = '&CASPT2'
    spinstring = 'Spin quantum number'
    statestring = 'Compute H0 matrices for state'
    refstring = 'Reference weight'
    stateindex = 5
    refindex = 2

    module = False
    correct_mult = False
    correct_state = False
    for i, line in enumerate(out):
        if modulestring in line:
            module = True
        elif 'Stop Module' in line:
            module = False
            correct_mult = False
            correct_state = False
        elif spinstring in line and module:
            spin = float(line.split()[3])
            if int(2 * spin) + 1 == mult:
                correct_mult = True
        elif statestring in line and module and correct_mult:
            if state == int(line.split()[stateindex]):
                correct_state = True
        elif refstring in line and module and correct_mult and correct_state:
            return float(line.split()[refindex])
    print('CASPT2 reference weight of state %i in mult %i not found!' % (state, mult))
    sys.exit(80)


# ======================================================================= #
def getcaspt2transform(out, mult):
    modulestring = '&CASPT2'
    spinstring = 'Spin quantum number'
    statestring = 'Number of CI roots used'
    matrixstring = 'Eigenvectors:'
    singlestatestring = 'This is a CASSCF or RASSCF reference function'
    stateindex = 5
    refindex = 2

    module = False
    correct_mult = False
    nstates = 0

    for i, line in enumerate(out):
        if modulestring in line:
            module = True
        elif 'Stop Module' in line:
            module = False
            correct_mult = False
        elif spinstring in line and module:
            spin = float(line.split()[3])
            if int(2 * spin) + 1 == mult:
                correct_mult = True
        elif statestring in line and module:
            nstates = int(line.split()[stateindex])
        elif singlestatestring in line and module and correct_mult:
            return [[1.]]
        elif matrixstring in line and module and correct_mult:
            t = [[0. for x in range(nstates)] for y in range(nstates)]
            for x in range(nstates):
                for y in range(nstates):
                    lineshift = i + y + 1 + x // 5 * (nstates + 1)
                    indexshift = x % 5
                    t[x][y] = float(out[lineshift].split()[indexshift])
            return t
    print('MS-CASPT2 transformation matrix in mult %i not found!' % (mult))
    sys.exit(81)


# ======================================================================= #
def verifyQMout(QMout, QMin, out):
    # checks whether a displacement calculation gave a sensible result
    # currently only checks for CASPT2 problems (reference weight)

    refweight_ratio = 0.80

    if QMin['method'] in [0, 3]:
        # CASSCF case
        pass
    elif QMin['method'] == 5 and QMin['pdft-functional'] != -1:
        pass
    elif QMin['method'] in [1, 2, 4, 5]:
        # SS-CASPT2 and MS-CASPT2 cases
        refs = []
        for istate in range(QMin['nmstates']):
            mult, state, ms = tuple(QMin['statemap'][istate + 1])
            if QMin['method'] in [1, 2]:
                refs.append(getcaspt2weight(out, mult, state))
            elif QMin['method'] in [3, 4, 5]:
                refs.append(1.)
            # print mult,state,refs[-1]

        # MS-CASPT2: get eigenvectors and transform
        if QMin['method'] == 2:
            offset = 0
            for imult, nstate in enumerate(QMin['states']):
                if nstate == 0:
                    continue
                mult = imult + 1
                for ims in range(mult):
                    t = getcaspt2transform(out, mult)
                    # pprint.pprint(t)
                    refslice = refs[offset:offset + nstate]
                    # print refslice
                    newref = [0. for i in range(nstate)]
                    for i in range(nstate):
                        for j in range(nstate):
                            newref[i] += refslice[j] * t[i][j] ** 2
                        refs[offset + i] = newref[i]
                    # print newref
                    offset += nstate
        for istate in range(QMin['nmstates']):
            mult, state, ms = tuple(QMin['statemap'][istate + 1])
            # print mult,state,refs[istate]

            # check the reference weights and set overlap to zero if not acceptable
            for istate in range(QMin['nmstates']):
                if refs[istate] < max(refs) * refweight_ratio:
                    QMout['overlap'][istate][istate] = complex(0., 0.)
                    # print('Set to zero:',istate)

    return QMout


# ======================================================================= #
def get_zeroQMout(QMin):
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    QMout = {}
    if 'h' in QMin or 'soc' in QMin:
        QMout['h'] = [[complex(0.0) for i in range(nmstates)] for j in range(nmstates)]
    if 'dm' in QMin:
        QMout['dm'] = [[[complex(0.0) for i in range(nmstates)] for j in range(nmstates)] for xyz in range(3)]
    if 'overlap' in QMin:
        QMout['overlap'] = [[complex(0.0) for i in range(nmstates)] for j in range(nmstates)]
    if 'grad' in QMin:
        QMout['grad'] = [[[0., 0., 0.] for i in range(natom)] for j in range(nmstates)]
    return QMout


# ======================================================================= #
def cleanupSCRATCH(SCRATCHDIR):
    ''''''
    if PRINT:
        print('===> Removing directory %s\n' % (SCRATCHDIR))
    try:
        if True:
            shutil.rmtree(SCRATCHDIR)
        else:
            print('not removing anything. SCRATCHDIR is %s' % SCRATCHDIR)
    except OSError:
        print('Could not remove directory %s' % (SCRATCHDIR))


# ======================================================================= #
def saveIntRun(WORKDIR, QMin):
    # Moves OrdInt, OneInt and Runfile files from WORKDIR to savedir
    # Need easier solution for PCAP: SBK
    # only 'master' branch is called.
    # SBK changed the following
    # TODO: change the sys.exit and the copy command!

    FromOrdInt = os.path.join(WORKDIR, 'MOLCAS.OrdInt')
    FromOneInt = os.path.join(WORKDIR, 'MOLCAS.OneInt')
    FromRunFile = os.path.join(WORKDIR, 'MOLCAS.RunFile')

    fromfile = []
    fromfile.append(FromOneInt)
    fromfile.append(FromOrdInt)
    fromfile.append(FromRunFile)

    for _fromfile in fromfile:
        if not os.path.isfile(_fromfile):
            print('File %s not found, cannot move to savedir!' % (_fromfile))
            sys.exit(82)
        fname = os.path.basename(_fromfile)
        tofile = os.path.join(QMin['savedir'], fname)
        if DEBUG:
            print('\nCopy:\t%s\n\t==>\n\t%s\n' % (_fromfile, tofile))
        shutil.copy(_fromfile, tofile)
        os.remove(_fromfile)
        # SBK added this in 06/22!


def savePCAPfiles(QMin, DEBUG=False):
    """
    Save relevant files from the scratch directory to the destination directory.
    Will need regenerating Projected CAP eta-trajectories. CAPMAT is stored.

    Parameters:
        QMin (dict): A dictionary containing configuration parameters.
        DEBUG (bool, optional): Debug mode flag. If True, entire scratch directory is copied. Default is False.
    """

    # Construct path to Main destination directory
    pathTO = os.path.join(QMin['savedir'], 'MOLCAS%s' % str(QMin['step'][0]))
    pathFROM = QMin['scratchdir']

    if DEBUG:
        try:
            os.makedirs(pathTO, exist_ok=True)
            shutil.copytree(pathFROM, pathTO, dirs_exist_ok=True)
        except Exception as e:
            print(f"\n===> An error occurred while copying the entire scratch directory: \n\t{e}")
            traceback.print_exc()
    else:
        if 'init' not in QMin:
            copy_subdirs = ['master', 'TRACK']
        else:
            copy_subdirs = ['master']

        # Iterate over subdirectories
        for subdir in os.listdir(pathFROM):
            if subdir in copy_subdirs:
                fromWORKDIR = os.path.join(pathFROM, subdir)
                toWORKDIR = os.path.join(pathTO, subdir)
                os.makedirs(toWORKDIR, exist_ok=True)
                for filename in os.listdir(fromWORKDIR):
                    fromfile = os.path.join(fromWORKDIR, filename)
                    tofile = os.path.join(toWORKDIR, filename)
                    try:
                        shutil.copy(fromfile, tofile)
                    except Exception as e:
                        print(f"\n ===> An error occurred (check scratch directory carefully!): \n\t{e}")
                        traceback.print_exc()
                    if DEBUG:
                        print(f'\nCopy:\t{fromfile}\n\t==>\n\t{tofile}\n')

    if 'track_wfoverlap' in QMin and 'init' not in QMin:
        WORKDIR_wfoverlap = os.path.join(pathFROM, 'TRACK')
        fromfile = os.path.join(WORKDIR_wfoverlap, 'dyson.out')
        tofile = os.path.join(pathTO, 'wfoverlap.out')
        shutil.copy(fromfile, tofile)
        if DEBUG:
            print(f'\nCopy:\t{fromfile}\n\t==>\n\t{tofile}\n')

        rmfiles = ['MOLCAS.RunFile', 'MOLCAS.OrdInt', 'MOLCAS.OneInt']
        for _delfile in rmfiles:
            WORKDIR_wfoverlap = os.path.join(pathTO, 'TRACK')
            delfile = os.path.join(WORKDIR_wfoverlap, _delfile)
            os.remove(delfile)
    return


def saveJobIphs(WORKDIR, QMin):
    # Moves JobIph files from WORKDIR to savedir

    # Need easier solution for PCAP: SBK
    # only 'master' branch is called.
    # SBK changed the following
    # TODO: chande the sys.exit and the copy command!
    if 'pcap' in QMin:
        fromfile = os.path.join(WORKDIR, 'MOLCAS.JobIph')
        if not os.path.isfile(fromfile):
            print('File %s not found, cannot move to savedir!' % (fromfile))
            sys.exit(82)
        tofile = os.path.join(QMin['savedir'], 'MOLCAS.JobIph')
        '''if DEBUG:
            print('\nCopy:\t%s\n\t==>\n\t%s\n' % (fromfile, tofile))'''
        print('\nCopy:\t%s\n\t==>\n\t%s\n' % (fromfile, tofile))  # SBK added!
        shutil.copy(fromfile, tofile)

        if 'scforb' in QMin:
            fromfile_extra = os.path.join(WORKDIR, 'MOLCAS.ScfOrb')
            if not os.path.isfile(fromfile_extra):
                print('File %s not found, cannot move to savedir!' % (fromfile_extra))
                sys.exit(821)
            tofile_extra = os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb')
            '''if DEBUG:
                print('\nCopy:\t%s\n\t==>\n\t%s\n' % (fromfile_extra, tofile_extra))'''
            print('\nCopy:\t%s\n\t==>\n\t%s\n' % (fromfile_extra, tofile_extra))  # SBK added!
            shutil.copy(fromfile_extra, tofile_extra)

        if 'track_wfoverlap' in QMin or 'rasorb' in QMin:  # Some extra files are needed
            fromFILE_wfoverlap = []
            fromFILE_wfoverlap.append(os.path.join(WORKDIR, 'MOLCAS.RasOrb'))
            for fromfile_extra in fromFILE_wfoverlap:
                if not os.path.isfile(fromfile_extra):
                    print('File %s not found, cannot move to savedir!' % (fromfile_extra))
                    sys.exit(822)
                tofile_extra = os.path.join(QMin['savedir'])
                if DEBUG:
                    print('\nCopy:\t%s\n\t==>\n\t%s\n' % (fromfile_extra, tofile_extra))
                shutil.copy(fromfile_extra, tofile_extra)


# ======================================================================= #


def moveJobIphs(QMin):
    # moves all relevant JobIph files in the savedir to old-JobIph files
    # deletes also all old .master files

    # SBK added this
    if 'pcap' in QMin:
        fromfile_dict = []
        fromfile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.JobIph'))
        if 'scforb' in QMin:
            fromfile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb'))

        tofile_dict = []
        tofile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.JobIph.old'))
        if 'scforb' in QMin:
            tofile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.ScfOrb.old'))
        if 'track_wfoverlap' in QMin or 'rasorb' in QMin:
            fromfile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.RasOrb'))
            tofile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.RasOrb.old'))

        if 'adaptive_track' in QMin:
            for _ct, _state in enumerate(QMin['template']['act_states']):
                if QMin['template']['update_ref'][_ct] == True:
                    fromfile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.JobIph'))
                    tofile_dict.append(os.path.join(QMin['savedir'], 'MOLCAS.JobIph.old.ref.%s' % (_ct + 1)))

        for i, fname in enumerate(fromfile_dict):
            if not os.path.isfile(fname):
                print('File %s not found, cannot move to OLD!' % (fname))
                sys.exit(830)
            if DEBUG:
                print('Copy:\t%s\n\t==>\n\t%s' % (fname, tofile_dict[i]))
            shutil.copy(fname, tofile_dict[i])

    ls = os.listdir(QMin['savedir'])
    for i in ls:
        if '.master' in i:
            rmfile = os.path.join(QMin['savedir'], i)
            os.remove(rmfile)


# ======================================================================= #


def stripWORKDIR(WORKDIR):
    ls = os.listdir(WORKDIR)
    # SBK added "'MOLCAS.JobIph'", "'MOLCAS.rassi.h5'"
    keep = ['MOLCAS.out',
            # 'MOLCAS.JobIph',
            'MOLCAS.rassi.h5',
            'MOLCAS\\.[1-9]\\.JobIph',
            'MOLCAS\\.[1-9]\\.RasOrb',
            'MOLCAS\\.[1-9]\\.molden']
    for ifile in ls:
        delete = True
        for k in keep:
            if containsstring(k, ifile):
                delete = False
        if delete:
            rmfile = os.path.join(WORKDIR, ifile)
            if not DEBUG:
                if os.path.isdir(rmfile):
                    cleanupSCRATCH(rmfile)
                else:
                    os.remove(rmfile)


# =============================================================================================== #
# =============================================================================================== #
# ===========================================  Dyson norms  ===================================== #
# =============================================================================================== #
# =============================================================================================== #

def decompose_csf(ms2, step):
    # ms2 is M_S value
    # step is step vector for CSF (e.g. 3333012021000)

    def powmin1(x):
        a = [1, -1]
        return a[x % 2]

    # calculate key numbers
    nopen = sum([i == 1 or i == 2 for i in step])
    nalpha = int(nopen / 2. + ms2)
    norb = len(step)

    # make reference determinant
    refdet = deepcopy(step)
    for i in range(len(refdet)):
        if refdet[i] == 1:
            refdet[i] = 2

    # get the b vector and the set of open shell orbitals
    bval = []
    openorbs = []
    b = 0
    for i in range(norb):
        if step[i] == 1:
            b += 1
        elif step[i] == 2:
            b -= 1
        bval.append(b)
        if refdet[i] == 2:
            openorbs.append(i)

    # loop over the possible determinants
    dets = {}
    # get all possible combinations of nalpha orbitals from the openorbs set
    for localpha in itertools.combinations(openorbs, nalpha):
        # make determinant string
        det = deepcopy(refdet)
        for i in localpha:
            det[i] = 1

        # get coefficient
        coeff = 1.
        sign = +1
        m2 = 0
        for k in range(norb):
            if step[k] == 1:
                m2 += powmin1(det[k] + 1)
                num = bval[k] + powmin1(det[k] + 1) * m2
                denom = 2. * bval[k]
                if num == 0.:
                    break
                coeff *= 1. * num / denom
            elif step[k] == 2:
                m2 += powmin1(det[k] - 1)
                num = bval[k] + 2 + powmin1(det[k]) * m2
                denom = 2. * (bval[k] + 2)
                sign *= powmin1(bval[k] + 2 - det[k])
                if num == 0.:
                    break
                coeff *= 1. * num / denom
            elif step[k] == 3:
                sign *= powmin1(bval[k])
                num = 1.

        # add determinant to dict if coefficient non-zero
        if num != 0.:
            dets[tuple(det)] = 1. * sign * math.sqrt(coeff)

    # pprint.pprint( dets)
    return dets


# ======================================================================= #


def get_determinants(out, mult):
    # first, find the correct RASSI output section for the given multiplicity
    modulestring = '&RASSI'
    spinstring = 'SPIN MULTIPLICITY:'
    stopstring = 'The following data are common to all the states'
    module = False
    jobiphmult = []
    for iline, line in enumerate(out):
        if modulestring in line:
            module = True
            jobiphmult = []
        elif module:
            if spinstring in line:
                jobiphmult.append(int(line.split()[-1]))
            if stopstring in line:
                if all(i == mult for i in jobiphmult):
                    break
                else:
                    module = False
    else:
        print('Determinants not found!', mult)
        print('No RASSI run for multiplicity %i found!' % (mult))
        sys.exit(84)

    # ndocc and nvirt
    ndocc = -1
    nvirt = -1
    while True:
        iline += 1
        line = out[iline]
        if ' INACTIVE ' in line:
            ndocc = int(line.split()[-1])
        if ' SECONDARY ' in line:
            nvirt = int(line.split()[-1])
        if ndocc != -1 and nvirt != -1:
            break

    # Get number of states
    while True:
        iline += 1
        line = out[iline]
        if 'Nr of states:' in line:
            nstates = int(line.split()[-1])
            break

    # Now start searching at iline, collecting all CI vectors
    # the dict has det-string tuples as keys and lists of float as values
    ci_vectors = {}
    statesstring = 'READCI called for state'
    stopstring = '****************************'
    finalstring = 'HAMILTONIAN MATRIX'
    done = set()

    finished = False
    while True:
        while True:
            iline += 1
            line = out[iline]
            if statesstring in line:
                state = int(line.split()[-1])
                break
            if finalstring in line:
                finished = True
                break
        if finished:
            break
        if state in done:
            continue
        done.add(state)
        iline += 13
        # TODO: verify now many lines need to be skipped! in newer versions might only be 10 or 11!
        while True:
            iline += 1
            line = out[iline]
            if stopstring in line:
                break
            s = line.split()
            if s == []:
                continue
            coef = float(s[-2])
            if coef == 0.:
                continue
            csf = s[-3]
            step = []
            for i in csf:
                if i == '2':
                    step.append(3)
                elif i == 'd':
                    step.append(2)
                elif i == 'u':
                    step.append(1)
                elif i == '0':
                    step.append(0)
            dets = decompose_csf((mult - 1) / 2., step)
            # add dets to ci_vectors
            for det in dets:
                if det in ci_vectors:
                    d = state - len(ci_vectors[det])
                    if d > 0:
                        ci_vectors[det].extend([0.] * d)
                    ci_vectors[det][state - 1] += coef * dets[det]
                else:
                    ci_vectors[det] = [0.] * state
                    ci_vectors[det][state - 1] += coef * dets[det]
    for det in ci_vectors.keys():
        d = nstates - len(ci_vectors[det])
        if d > 0:
            ci_vectors[det].extend([0.] * d)
        # if all( i==0. for i in ci_vectors[det] ):
        # del ci_vectors[det]
    ci_vectors['ndocc'] = ndocc
    ci_vectors['nvirt'] = nvirt
    # pprint.pprint( ci_vectors)
    return ci_vectors


# ======================================================================= #


def format_ci_vectors(ci_vectors):
    # get nstates, norb and ndets
    for key in ci_vectors:
        if key != 'ndocc' and key != 'nvirt':
            nstates = len(ci_vectors[key])
            norb = len(key)
    ndets = len(ci_vectors) - 2
    ndocc = ci_vectors['ndocc']
    nvirt = ci_vectors['nvirt']

    # sort determinant strings
    dets = []
    for key in ci_vectors:
        if key != 'ndocc' and key != 'nvirt':
            dets.append(key)
    dets.sort(reverse=True)

    string = '%i %i %i\n' % (nstates, norb + ndocc + nvirt, ndets)
    for det in dets:
        for i in range(ndocc):
            string += 'd'
        for o in det:
            if o == 0:
                string += 'e'
            elif o == 1:
                string += 'a'
            elif o == 2:
                string += 'b'
            elif o == 3:
                string += 'd'
        for i in range(nvirt):
            string += 'e'
        for c in ci_vectors[det]:
            string += ' %16.12f ' % c
        string += '\n'
    return string


def format_ci_diag_vectors(ci_vectors, lvec, rvec, dtype='real'):
    for key in ci_vectors:
        if key != 'ndocc' and key != 'nvirt':
            nstates = len(ci_vectors[key])
            norb = len(key)
    ndets = len(ci_vectors) - 2
    ndocc = ci_vectors['ndocc']
    nvirt = ci_vectors['nvirt']
    nroots = len(lvec)

    # sort determinant strings
    dets = []
    for key in ci_vectors:
        if key != 'ndocc' and key != 'nvirt':
            dets.append(key)
    dets.sort(reverse=True)

    ci_vectors_diag = []
    for det in dets:
        for c in ci_vectors[det]:
            d_ci = []
            for k in range(nroots):
                d_ci_store = 0.0
                for i in range(nroots):
                    d_ci_store += lvec.T[k, i] * c[i] * rvec[i, k]
                if dtype.lower() == 'imag':
                    d_ci.append(d_ci_store.imag)
                else:
                    d_ci.append(d_ci_store.real)
            ci_vectors_diag.append(d_ci)

    string = '%i %i %i\n' % (nstates, norb + ndocc + nvirt, ndets)
    for det in dets:
        for i in range(ndocc):
            string += 'd'
        for o in det:
            if o == 0:
                string += 'e'
            elif o == 1:
                string += 'a'
            elif o == 2:
                string += 'b'
            elif o == 3:
                string += 'd'
        for i in range(nvirt):
            string += 'e'
        for c in ci_vectors_diag[det]:
            string += ' %16.12f ' % c
        string += '\n'
    return string


def print_civecs_diag(nroots, lvec, rvec, finp, fout, dtype='real'):
    with open(finp, 'r') as f:
        flines = f.readlines()
    civectors = []
    for idx, line in enumerate(flines[1:]):
        civectors.append([float(i) for i in line.split()[1:]])

    civectors_diag = []
    for idx, civecs in enumerate(civectors):
        d_ci = []
        for k in range(nroots):
            d_ci_store = 0.0
            for i in range(nroots):
                d_ci_store += lvec.T[k, i] * civecs[i] * rvec[i, k]
            d_ci.append(d_ci_store)
        civectors_diag.append(d_ci)

    detsstring = ''

    icivec = 0
    for idx, line in enumerate(flines):
        if len(line) > nroots:
            if dtype.lower() == 'imag':
                cistring = "".join(' %16.12f ' % (i.imag) for i in civectors_diag[icivec])
            else:
                cistring = "".join(' %16.12f ' % (i.real) for i in civectors_diag[icivec])
            icivec += 1
            detsstring_line = f"{line.split()[0]} {cistring}"
            detsstring += '%s\n' % detsstring_line
        else:
            detsstring += '%s' % line

    with open(fout, 'w') as fdiag:
        fdiag.write(detsstring)


# ======================================================================= #
def runWFOVERLAPS(WORKDIR, wfoverlaps, memory=1000, ncpu=1):
    prevdir = os.getcwd()
    os.chdir(WORKDIR)
    string = wfoverlaps + ' -m %i' % (memory) + ' -f dyson.in'
    stdoutfile = open(os.path.join(WORKDIR, 'dyson.out'), 'w')
    stderrfile = open(os.path.join(WORKDIR, 'dyson.err'), 'w')
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


# =============================================================================================== #
# =============================================================================================== #
# =========================================== Main routine  ===================================== #
# =============================================================================================== #
# =============================================================================================== #

# ========================== Main Code =============================== #
def main():
    # Retrieve PRINT and DEBUG
    try:
        envPRINT = os.getenv('SH2CAS_PRINT')
        if envPRINT and envPRINT.lower() == 'false':
            global PRINT
            PRINT = False
        envDEBUG = os.getenv('SH2CAS_DEBUG')
        if envDEBUG and envDEBUG.lower() == 'true':
            global DEBUG
            DEBUG = True
    except ValueError:
        print('PRINT or DEBUG environment variables do not evaluate to numerical values!')
        sys.exit(90)

    # Process Command line arguments
    if len(sys.argv) != 2:
        print('Usage:\n./SHARC_MOLCAS_OPENCAP.py <QMin>\n')
        print('version:', version)
        print('date:', versiondate)
        sys.exit(91)
    QMinfilename = sys.argv[1]

    # Print header
    printheader()

    # Read QMinfile
    QMin = readQMin(QMinfilename)

    # make list of jobs
    QMin, joblist = generate_joblist(QMin)

    # run all MOLCAS jobs
    errorcodes = runjobs(joblist, QMin)

    # get output
    if 'pcap' in QMin:
        QMoutall = collectOutputsPCAP(joblist, QMin, errorcodes)

    # format final output
    QMout = arrangeQMout(QMin, QMoutall, None)

    # Measure time
    runtime = measuretime()
    QMout['runtime'] = runtime

    # Write QMout
    writeQMout(QMin, QMout, QMinfilename)

    # SBK added this
    if 'pcap' in QMin:
        savePCAPfiles(QMin, DEBUG=False)

    # Remove Scratchfiles from SCRATCHDIR
    if not DEBUG:
        cleanupSCRATCH(QMin['scratchdir'])
        if 'cleanup' in QMin:
            cleanupSCRATCH(QMin['savedir'])

    #
    if PRINT or DEBUG:
        print('#================ END ================#')


if __name__ == '__main__':
    main()

# kate: indent-width 4
