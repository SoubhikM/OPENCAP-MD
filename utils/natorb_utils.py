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

'''
Some utility functions that generates natural orbitals in DIAG basis
'''

import numpy as np
import pyscf.tools.molden
import scipy.linalg as LA
from functools import reduce
from pyscf.gto.basis import parse_gaussian
from pyscf import gto

ANG_TO_BOHR = 1.8897259886

class dm_diag:
    '''
    Generates 1-RDMs in DIAG basis
    '''
    def __init__(self, pcOBJ):
        self.pcapobj = pcOBJ

    def _dm_AO_total(self, k, l):
        return  self.pcapobj.get_density(k, l, False) + self.pcapobj.get_density(k, l, True)

    def generate_DM_diag(self, Rvec, LVec):
        nroots = len(LVec)
        DM_diag = {}
        for res_state1 in range(nroots):
            for res_state2 in range(nroots):
                _density = 0.0
                for k in range(nroots):
                    for l in range(nroots):
                        _density += LVec.T[res_state1, k] * self._dm_AO_total(k, l) * Rvec[l, res_state2]
                DM_diag[res_state1, res_state2] = _density.copy()
        return DM_diag

class natorb:
    '''A class that calculates natural orbitals'''
    def __init__(self, pcOBJ, ovlp, RotMat, geomfile, QMin, BASFILE, fMOLDEN):
        self._basfile = BASFILE
        self.fMOLDEN = fMOLDEN
        self.rotmat = RotMat
        self.nroots = int(QMin['nstates'])
        self.projcap = pcOBJ
        self.ao_ovlp = ovlp
        self.geomfile = geomfile
        self.res_state = int(QMin['act_state'][0])-1

    def orthoMO(self, C, S):
        '''
        Adapted from MOLCAS, normalizing C to C', so that C'^TSC' is Identity matrix.
        '''
        W = reduce(np.dot, (C.T, S, C))
        W_sqrt = LA.sqrtm(W)
        W_inv_sqrt = LA.inv(W_sqrt)
        Cprime = reduce(np.dot, (C, W_inv_sqrt))
        return Cprime

    def atom_labelfree(self, xyzdict):
        unique_labels = []
        for label, _, _, _ in xyzdict:
            unique_label = ''.join(filter(str.isalpha, label))
            if unique_label not in unique_labels:
                unique_labels.append(unique_label)
        return unique_labels

    def _xyz4pyscf(self, fname):
        with open(fname, 'r') as file:
            lines = file.readlines()
        coords = []
        for _idx, line in enumerate(lines):
            label, _, x, y, z, _ = line.split()
            coords.append((label, float(x)/ANG_TO_BOHR, float(y)/ANG_TO_BOHR, float(z)/ANG_TO_BOHR))
        return coords

    def _generate_NOs(self, S, nroots, DM_diag):
        noons_all = []
        natorbs_all = []
        for i in range(nroots):
            A = reduce(np.dot, (DM_diag[i, i], S))
            w, v = LA.eig(A)
            v = self.orthoMO(v, S)
            _des_idx = np.argsort(w.real)[::-1]
            noons = w[_des_idx].real
            natorbs = v[:, _des_idx].real
            noons_all.append(noons)
            natorbs_all.append(natorbs)
        return noons_all, natorbs_all

    def print_natorb_molden(self, allstate=False):
        atomdict = self._xyz4pyscf(self.geomfile)
        atomlabels = self.atom_labelfree(atomdict)

        gbas_dict = {}
        for _, atomlabel in enumerate(atomlabels):
            gbas_dict[atomlabel] = parse_gaussian.load(self._basfile, atomlabel)

        Reigvc = self.rotmat['Reigvc']
        Leigvc = self.rotmat['Leigvc']
        DM_diag = dm_diag(self.projcap).generate_DM_diag(Reigvc, Leigvc)

        for i in range(self.nroots):
            _nocc_wf, _natorb_wf = self._generate_NOs(self.ao_ovlp, self.nroots, DM_diag)

        mymol = gto.M(atom=atomdict,basis=gbas_dict, symmetry=False, verbose=0)

        if allstate:
            for istate in range(self.nroots):
                trimmed_fmolden = self.fMOLDEN.rsplit('.', 2)[0]
                modified_fmolden = f"{trimmed_fmolden}.state{istate+1}.sp"
                pyscf.tools.molden.from_mo(mymol, filename=modified_fmolden, mo_coeff=_natorb_wf[istate],
                                           ene=np.arange(0, mymol.nao, 1, dtype=float),
                                           occ=_nocc_wf[istate])
        else:
            pyscf.tools.molden.from_mo(mymol, filename=self.fMOLDEN, mo_coeff=_natorb_wf[self.res_state],
                                   ene=np.arange(0, mymol.nao, 1, dtype=float),
                                   occ=_nocc_wf[self.res_state])
