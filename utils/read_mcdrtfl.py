#!/usr/bin/env python3

#******************************************
#
#    Modified SHARC/WFOVERLAP Code
#
#    Copyright (c) 2024 Soubhik Mondal
#
#    This file is based on SHARC and WFOVERLAP, originally developed
#    at the University of Vienna and authored by S. Mai.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#******************************************

'''
This program extracts and prints Configuration State Functions (CSFs) from
MCSCF calculations in COLUMBUS.

It builds on functions from the SHARC-MOLCAS interface and the read_rassi.py script
from the SHARC/WFOVERLAP program (originally by S. Mai).
(https://github.com/sharc-md/sharc/blob/main/wfoverlap/scripts/read_rassi.py)

Functions directly adapted with minor modifications:
- decompose_csf
- get_determinants
- format_ci_vectors

Modifications were made for compatibility with COLUMBUS outputs and extended functionality.
'''


import copy
import math
import itertools
import subprocess as sp
import os, datetime, sys


class mcdrtfl_dets:

    def read_orb_info(self, path):
        with open(os.path.join(path, 'mcscfls'),'r') as f:
            mcscfls = f.readlines()

        with open(os.path.join(path, 'mcdrtfl'), 'r') as f:
            mcdrtfl = f.readlines()

        for idx, mcline in enumerate(mcdrtfl):
            if 'info' in mcline:
                orbs = [int(i) for i in mcdrtfl[idx + 1].split()[:2]]
                self.ndocc, self.nact = orbs
                self.ncsf = int(mcdrtfl[idx + 1].split()[-1])
                break
        for idx, mcline in enumerate(mcscfls):
            if 'Total number of basis functions:' in mcline:
                self.nmo = int(mcline.split()[-1])
                break

        self.nvirt = self.nmo-self.ndocc-self.nact

    def __init__(self, path, debug=False, ms=0.5,  mem=2000, columbus=os.environ['COLUMBUS']):
        self.debug = debug
        self.columbus = columbus
        self.ms = ms
        self.mem = mem
        self.path = path
        self.read_orb_info(self.path)


    def decompose_csf(self, ms2, step):
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
        refdet = copy.deepcopy(step)
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
            det = copy.deepcopy(refdet)
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


    def get_determinants(self, csf_dicts_allstate, mult):
        nstates = len(csf_dicts_allstate)
        ci_vectors = {}
        for istate, csf_dict in enumerate(csf_dicts_allstate):

            state = istate + 1
            csfs = csf_dict[0]
            coeffs = csf_dict[1]

            for idx, (csf, coef) in enumerate(zip(csfs, coeffs)):

                step = []
                for i in csf:
                    if i == '3':
                        step.append(3)
                    elif i == '2':
                        step.append(2)
                    elif i == '1':
                        step.append(1)
                    elif i == '0':
                        step.append(0)
                dets = self.decompose_csf((mult - 1) / 2., step)

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
        ci_vectors['ndocc'] = self.ndocc
        ci_vectors['nvirt'] = self.nvirt
        # pprint.pprint( ci_vectors)
        return ci_vectors


    def extract_csf_elems(self, mcpcls):
        csfsec = False
        csf_data = []
        coeff_data = []

        for idx, line in enumerate(mcpcls):
            if "csf" in line:
                csfsec = True

            if csfsec and len(line.split()) == 4 and line.split()[0].isdigit():
                csf_data.append(line.split()[-1])
                coeff_data.append(float(line.split()[1]))

        csf_dict = []
        csf_dict.append(csf_data)
        csf_dict.append(coeff_data)

        return csf_dict


    def format_ci_vectors(self, ci_vectors):
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

    def call_mcpc(self, istate):
        """
        Call mcpc.x and analyse the information on the fly.
        """

        command = ["%s/mcpc.x" % self.columbus, "-m", "%i" % self.mem]
        istart = 1
        iend = self.ncsf

        mcpcinstr = "1\n"  # initialize determinant print out for DRT1
        mcpcinstr += "%i\n" % istate  # read the state
        mcpcinstr += "3\n"  # print the csfs
        mcpcinstr += "%i %i\n" % (istart, iend)  # first and last CSF to print
        mcpcinstr += "0,0\n"  # Get out
        mcpcinstr += "0\n"  # Terminate
        print("%s/mcpc.x for state %i, CSFs %i to %i" % (self.columbus, istate, istart, iend))

        #Run it then
        starttime = datetime.datetime.now()
        sys.stdout.write('\t%s' % (starttime))
        mcpc = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
        mcpcls, mcpcerr = mcpc.communicate(mcpcinstr.encode('utf-8'))

        if debug:
            open('pymcpcin.st%i' % (istate), 'w').write(mcpcinstr)
            open('pymcpcls.st%i' % (istate), 'w').write(mcpcls.decode('utf-8'))
        if not 'end of mcpc' in mcpcerr.decode('utf-8'):
            print(" ERROR in mcpc.x during determinant generation!")
            print("\n Standard error:\n", mcpcerr, 'Exit code:', mcpc.returncode)
            sys.exit(20)
        if debug:
            print(" mcpc.x finished succesfully")
        endtime = datetime.datetime.now()

        sys.stdout.write('\t%s\t\tRuntime: %s\n\n' % (endtime, endtime - starttime))

        mcpcls_lines = mcpcls.decode('utf-8').splitlines()

        return mcpcls_lines

    def writefile(self, filename, content):
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

    def write_det_file(self, nstates, filedets):
        csf_dicts_allstate = []
        for istate in range(nstates):
            #In mcpc.x call, it starts at state 1 (fortran index)
            mcpcls = self.call_mcpc(istate+1)
            csf_dict = self.extract_csf_elems(mcpcls)
            csf_dicts_allstate.append(csf_dict)

        ci_vectors = self.get_determinants(csf_dicts_allstate, 2.*self.ms+1.)
        ci_vectorsstr = self.format_ci_vectors(ci_vectors)
        self.writefile(filedets, ci_vectorsstr)


if __name__=="__main__":

    '''
    Need to run in the directory containing restart, mcdrtfl, mcscfls files generated from MCSCF run.
    '''
    wname = 'dets'
    nstates = None
    debug = False
    ms = 0.5
    mem = 1000

    args = sys.argv[1:]
    if len(args) == 0:
        print("Enter at least one argument!\n")
        sys.exit()

    while (len(args) >= 1):
        arg = args.pop(0)
        if arg == '-debug':
            debug = True
        elif arg == '-ms':
            ms = float(args.pop(0))
        elif arg == '-m':
            mem = int(args.pop(0))
        elif arg == '-o':
            wname = args.pop(0)
        else:
            if nstates == None:
                nstates = int(arg)

    pathRUN = os.getcwd()
    filedets = os.path.join(pathRUN, wname)

    mcdrtfl_dets(pathRUN, debug=debug, ms=ms, mem=mem).write_det_file(nstates, filedets)