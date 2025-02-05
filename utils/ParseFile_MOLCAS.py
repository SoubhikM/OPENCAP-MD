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

import copy
import re
#import warnings
import numpy
import warnings

class ParseFile:
    def __init__(self, flines, nroots, natom, ghost=True):
        self.f = flines
        self.nroots = nroots
        self.Natom = natom
        self.ghost = ghost

        # Adjust atom count if ghost atoms are included
        self.Natom = self._adjust_atom_count()

    def _adjust_atom_count(self):
        """Adjust atom count based on ghost atom inclusion."""
        if self.ghost:
            return self.Natom + 1  # Add ghost atom count
        return self.Natom

    def parseNAC(self, _state1, _state2):
        #This gives us d_JI : I is _state1
        #                   : J is _state2
        _state_nac={}
        lookSTR  = re.compile(r'Lagrangian multipliers are calculated for states no.+\s%s/\s+%s\s'%(_state1, _state2))
        for _ct, line in enumerate(self.f):
            if lookSTR.search(line):
                RemLines = self.f[_ct+1:]
                for ct, remline in enumerate(RemLines):
                    if "Lagrangian multipliers are calculated for states no." in remline:
                        warnings.warn("NAC not found for (%s,%s) pair."%(_state1, _state2))
                        pass
                    elif "Total derivative coupling" in remline:
                        _header = [item.replace(")", " ") for item in remline.split()]
                        factor = 1.0
                        if "divided by" in remline:
                            for _, str in enumerate (_header):
                                try:
                                    factor = float(str)
                                except:
                                    continue

                        NAClines = [item.split() for item in RemLines[ct+8:ct+8+self.Natom]]
                        NAClines = [_modlist for _modlist in NAClines if not re.match(r'^X\d*$', _modlist[0])]
                        #print(NAClines)
                        for _atomct, str in enumerate (NAClines):
                            NACatom = [float(ix)*factor for ix in str[1:]]
                            _state_nac[_atomct] = { 'x': NACatom[0],
                                                    'y': NACatom[1],
                                                    'z': NACatom[2] }
                        break

        return _state_nac


    def parseGRAD(self, _state):
        _state_grad={}
        lookSTR = re.compile(r'Lagrangian multipliers are calculated for state no.+\s+%s\s'%(_state))
        for _ct, line in enumerate(self.f):
            if lookSTR.search(line):
                RemLines = self.f[_ct+1:]
                for ct, remline in enumerate(RemLines):
                    if "Lagrangian multipliers are calculated for state no." in remline:
                        warnings.warn("Gradient not found for state : %s"%_state)
                        pass
                    elif "Molecular gradients" in remline:
                        _header = [item.replace(")", " ") for item in remline.split()]
                        factor = 1.0
                        if "divided by" in remline:
                            for _, str in enumerate (_header):
                                try:
                                    factor = float(str)
                                except:
                                    continue

                        GRADlines = [item.split() for item in RemLines[ct+8:ct+8+self.Natom]]
                        #print(GRADlines)
                        GRADlines = [_modlist for _modlist in GRADlines if not re.match(r'^X\d*$', _modlist[0])]
                        #print(GRADlines)
                        for _atomct, str in enumerate (GRADlines):
                            GRADatom = [float(ix)*factor for ix in str[1:]]
                            _state_grad[_atomct] = { 'x': GRADatom[0],
                                                     'y': GRADatom[1],
                                                     'z': GRADatom[2] }
                        break
        return _state_grad

    #Not needed
    '''
    def parseH0(self, _state):
        if _state >9:
            LOOKstr = "::    RASSI State   %s     Total energy: "%_state
        else:
            LOOKstr = "::    RASSI State    %s     Total energy: "%_state

        for _ct, line in enumerate(self.f):
            if LOOKstr in line:
                energy = float(line.split()[-1])
        return energy
    '''

    def GRADmat(self):
        _grad={}
        for _state in range (self.nroots):
            _grad[_state] = self.parseGRAD(_state+1)

        return _grad

    def NACmat(self):
        _nac={}
        for _state1 in range (self.nroots):
            for _state2 in range (_state1+1, self.nroots):
                _nac[_state1, _state2] = self.parseNAC(_state1+1, _state2+1)
        return _nac
    #Not needed
    '''
    def getH0(self):
        _energy={}
        for _state in range (self.nroots):
            _energy[_state] = self.parseH0(_state+1)

        return _energy
    '''

    def grad_mat(self):
        _grad = self.GRADmat()
        _nac = self.NACmat()
        #H0 = self.getH0()

        grad_all_diag = {}
        grad_all_offdiag = {}
	
        natom_no_ghost=self.Natom-1 if self.ghost else self.Natom #SBK: subtracting one ghost atom, very shoddy solution #TODO
        for iatom in range(natom_no_ghost): 
            grad = {}
            nac = {}
            for idir in {'x', 'y', 'z'}:
                _grad_store = numpy.zeros([self.nroots, self.nroots])
                _nac_store = numpy.zeros([self.nroots, self.nroots])
                for _state1 in range (self.nroots):
                    try:
                        _grad_store[_state1, _state1] = _grad[_state1][iatom][idir]
                    except:
                        warnings.warn("Warning: Gradient for state %s ignored.\nMost likely the job crashed or removed by screening!\n"%(_state1+1))
                        pass
                    for _state2 in range (_state1+1, self.nroots):
                        # Save F_IJ : I is _state1
                        #           : J is _state2
                        try:
                            _nac_store[_state1, _state2] = -1.0*_nac[_state1, _state2][iatom][idir]
                                                    #* (H0[_state2]-H0[_state1])
                            _nac_store[_state2, _state1] = -1.0*_nac_store[_state1, _state2]
                        except:
                            warnings.warn("Warning: Derivative coupling for [%s, %s] pair ignored. \nMost likely the job crashed or removed by screening!\n"%(_state1+1, _state2+1))
                            pass
                grad[idir] = copy.deepcopy(_grad_store)
                nac[idir] = copy.deepcopy(_nac_store)

            grad_all_diag[iatom] = {'x': grad['x'],
                                    'y': grad['y'],
                                    'z': grad['z']}
            grad_all_offdiag[iatom] = {'x': nac['x'],
                                       'y': nac['y'],
                                       'z': nac['z']}

        return grad_all_diag, grad_all_offdiag
