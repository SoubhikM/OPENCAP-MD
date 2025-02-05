#!/usr/bin/env python3

# ******************************************
#
#    Modified Version of SHARC Program Suite
#
#    Based on SHARC (https://sharc-md.org/)
#    Originally Copyright (c) 2023 University of Vienna
#
#    This modified version has undergone some minor changes by
#    Soubhik Mondal at Boston University
#
#    This file is distributed under the GNU General Public License v3 (GPLv3).
#    You may redistribute and modify it under the same license terms.
#
#    SHARC is licensed under GPL v3.
#    For license details, see <http://www.gnu.org/licenses/>.
#
# ******************************************

"""
This program reads a civfl file in Columbus format using cipc.x and converts it to the native ASCII format
used by cioverlap.x

Added capability to print CSFs when symmetry is on.

New functions added:
parse_ciudgls, extorb_from_sym
Need ciudgls file in addition to the usual files to run this program
"""


import subprocess as sp
import os, sys, datetime

class civfl_ana:
    def __init__(self, maxsqnorm=1.0, debug=False, columbus=os.environ['COLUMBUS']):
        self.det_dict = {}  # dictionary with determinant strings and cicoefficient information
        self.nmot = -1  # number of MOs
        self.niot = -1  # number of internal orbs
        self.nfct = -1  # number of frozen orbs
        self.nfvt = -1  # number of frozen virtuals
        self.ncsf = -1
        self.maxsqnorm = maxsqnorm
        self.sqcinorms = {}  # CI-norms
        self.debug = debug
        self.columbus = columbus

    # ================================================== #
    def call_cipc(self, istate, ms="0", csfbuf=200000, mem=1000):
        """
        Call cipc.x and analyse the information on the fly.
        The input is batched to limit the amount of memory used.
        The batch size is controlled by csfbuf.
        """
        command = ["%s/cipc.x" % self.columbus, "-m", "%i" % mem]
        istart = 1
        maxiter = 20000
        for i in range(maxiter):
            iend = istart + csfbuf
            cipstr = "2\n4\n1\n%s\n0\n" % ms  # initialize determinant print out
            cipstr += "7\n5\n%i\n0\n" % istate  # read the coefficients
#            cipstr += "1\n%i %i\n0/\n" % (istart, iend)  # first and last CSF to print
            cipstr += "3\n%i %i\n0/\n" % (istart, iend)  # first and last CSF to print, printing in csf order(3)
            cipstr += "0\n"  # finish

            starttime = datetime.datetime.now()
            if self.debug:
                sys.stdout.write("\nSTART:  %s/cipc.x for state %i, CSFs %i to %i\n" % (self.columbus, istate, istart, iend))
                sys.stdout.write('\t%s' % (starttime))
                sys.stdout.flush()

            cipx = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
            cipout, ciperr = cipx.communicate(cipstr.encode('utf-8'))
            cipout = cipout.decode('utf-8') if cipout else ""
            ciperr = ciperr.decode('utf-8') if ciperr else ""
            if not 'end of cipc' in ciperr:
                sys.stdout.write(" ERROR in cipc.x during determinant generation!")
                sys.stdout.write("\n Standard error:\n" + ciperr + " Exit code: " + str(cipx.returncode))
                sys.stdout.flush()
                sys.exit(20)
            self.read_cipinfo(istate, cipout)
            if self.debug:
                open('pycipcin.st%i' % (istate), 'w').write(cipstr)
                open('pycipcls.st%i' % (istate), 'w').write(cipout)

            endtime = datetime.datetime.now()
            if self.debug:
                sys.stdout.write('\n FINISH: \t%s\t\tRuntime: %s\n\n' % (endtime, endtime - starttime))
                sys.stdout.flush()

            if (iend > self.ncsf) or (self.sqcinorms[istate] > self.maxsqnorm):
                break
            else:
                istart = iend + 1
        else:
            print('%i CSFs read, maxiter reached.' % (csfbuf * maxiter))
            sys.exit(41)
    # ================================================== #
    def parse_ciudgls(self, fname, niot):
        '''
        For symmetry True calculations: Order of internal orbitals and external orbitals' symmetry

        :param fname: str, ciudgls filename
        :param niot: int, No of internal orbitals
        :return: list, internal and external orbitals with symmetry elements
        '''
        with open(fname, 'r') as fn:
            fciudgls = fn.readlines()
        levsym = []
        slabel = []
        for idx, line in enumerate(fciudgls):
            if 'levsym(*)' in line:
                for idxx, linex in enumerate(fciudgls[idx + 1:]):
                    for j in linex.split():
                        try:
                            levsym.append(int(j))
                        except:
                            pass
                    if '*' in fciudgls[idx:][idxx + 1]:
                        break
            elif 'slabel(*)' in line:
                for ilabel in (line.split()[2:]):
                    slabel.append(ilabel.lower())

        _orb_int = []
        _orb_ext = []
        for idx, iorb in enumerate(levsym[::-1]):
            if idx < niot:
                _orb_int.append(slabel[iorb - 1])
            else:
                _orb_ext.append(slabel[iorb - 1])
        return _orb_int[::-1], _orb_ext[::-1]

    def extorb_from_sym(self, ext_sym, ext1, orbint, orbext):
        all_orbs = []
        all_orbs.extend(orbint)
        all_orbs.extend(orbext)
        idx_sym = [i + 1 for i, orbital in enumerate(all_orbs) if orbital == ext_sym.lower()]
        return idx_sym[ext1 - 1]

    # ================================================== #
    def read_cipinfo(self, istate, fstring):
        if istate in self.sqcinorms:
            sqcinorm = self.sqcinorms[istate]
        else:
            sqcinorm = 0.
        flist = fstring.split("\n")
        flist.reverse()
        csfsec = False
        n_ext = 0;
        ext1 = 0;
        ext2 = 0
        while (True):
            line = flist.pop()
            if 'drt header information' in line:
                line = flist.pop()
                line = flist.pop()
                words = line.replace('=', ' ').split()
                self.nmot = int(words[1])
                self.niot = int(words[3])
                self.nfct = int(words[5])
                self.nfvt = int(words[7])
                # print 'Orbital information parsed:'
                # SBK added
                ciudgls = "./ciudgls"
                self.orbint, self.orbext = self.parse_ciudgls(ciudgls, self.niot)
                #
            elif 'ncsft:' in line:
                self.ncsf = int(line.split()[-1])
            if 'indcsf' in line:
                csfsec = True
                line = flist.pop()
                continue
            elif ('csfs were printed in this range' in line):
                break
            if not csfsec: continue
            if len(line) == 0:
                break
            words = line.split()
            if ('idet' in line):
                if sqcinorm > self.maxsqnorm:
                    break
                else:
                    continue
            elif (len(words) > 3):
                wtype = words[3]
                n_ext_el = {'z*': 0, 'z': 0, 'y': 1, 'x': 2, 'w': 2}[wtype]
                n_exto = line.count(':')  # number of external orbitals
                ext1 = ext2 = -1
                rwords = line.replace(':', ' ').split()
                if n_exto == 1:
                    ext1parse = int(rwords[-2])
                    # SBK ADDED this following lines
                    symtype = rwords[-3].lower()
                    ext1 = self.extorb_from_sym(symtype, ext1parse, self.orbint, self.orbext)
                    # End of addition
                elif n_exto == 2:
                    ext1 = int(rwords[-4])
                    ext2 = int(rwords[-2])
                CSFstr = words[-1]
                nel = 2 * CSFstr.count('3') + CSFstr.count('1') + CSFstr.count('2')
            else:
                coeff = float(words[1])
                # For even electron systems there is phase change if one external
                #   electron is moved from the front (where it is in the DRT)
                #   to the back (where we expect it to be).
                # With two external electrons, this cancels out.
                # For odd electron systems, the phases should never change.
                if (nel % 2 == 0) and (n_ext_el == 1): coeff = -coeff
                # if n_ext_el==1: coeff = -coeff
                det = self.det_string(words[-1], n_exto, ext1, ext2)
                # print(nel, coeff, det, words[-1], n_exto, ext1, ext2)
                if self.debug:
                    print("%25s -> %s: % .10f" % (words[-1], det, coeff))
                if det in self.det_dict:
                    if istate in self.det_dict[det]:
                        self.det_dict[det][istate] += coeff
                    else:
                        self.det_dict[det][istate] = coeff
                else:
                    self.det_dict[det] = {istate: coeff}
                sqcinorm += coeff ** 2
        self.sqcinorms[istate] = sqcinorm

    # ================================================== #
    def det_string(self, cipstr, n_ext, ext1, ext2):
        retstr = self.nfct * 'd'
        retstr += self.det_labels(cipstr[n_ext:])
        for iorb in range(self.nfct + self.niot + 1, self.nmot + 1):
            if iorb == ext1:
                retstr += self.det_labels(cipstr[0])
            elif iorb == ext2:
                retstr += self.det_labels(cipstr[1])
            else:
                retstr += 'e'
        return retstr

    # ================================================== #
    def det_labels(self, cipstr):
        return cipstr.replace('#', 'd').replace('+', 'a').replace('-', 'b').replace('.', 'e')

    # ================================================== #
    def sort_key(self, key):
        """
        For specifying the sorting order of the determinants.
        """
        return key.replace('d', '0').replace('a', '1').replace('b', '1')

    # ================================================== #
    def sort_key2(self, key):
        """
        For specifying the sorting order of the determinants.
        """
        return key.replace('d', '0').replace('a', '0').replace('b', '1').replace('e', '1')

    # ================================================== #
    def write_det_file(self, nstate, wname='dets', wform=' % 14.10f'):
        wf = open(wname, 'w')
        wf.write("%i %i %i\n" % (nstate, self.nmot, len(self.det_dict)))
        for det in sorted(sorted(self.det_dict, key=self.sort_key2), key=self.sort_key):
            wf.write(det)
            for istate in range(1, nstate + 1):
                try:
                    coeff = self.det_dict[det][istate]
                except KeyError:
                    coeff = 0.
                wf.write(wform % coeff)
            wf.write('\n')
        wf.close()


if __name__ == '__main__':
    print("read_civfl_w_sym.py <nstate> [<maxsqnorm>]")
    print("   command line options: -debug, -ms <ms>, -m <mem (MB)>, -o <det_file>\n")

    debug = False
    nstate = None
    maxsqnorm = 2.0
    mem = 1000
    ms = 0
    wname = 'dets'

    args = sys.argv[1:]
    if len(args) == 0:
        print("Enter at least one argument!\n")
        sys.exit()

    while (len(args) >= 1):
        arg = args.pop(0)
        if arg == '-debug':
            debug = True
        elif arg == '-ms':
            ms = args.pop(0)
        elif arg == '-m':
            mem = int(args.pop(0))
        elif arg == '-o':
            wname = args.pop(0)
        else:
            if nstate == None:
                nstate = int(arg)
            else:
                maxsqnorm = float(arg)

    ca = civfl_ana(maxsqnorm, debug)

    for istate in range(1, nstate + 1):
        ca.call_cipc(istate, ms=ms, mem=mem)
        # ca.read_cipcls(istate)

    ca.write_det_file(nstate, wname=wname)