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

MASSES = {'H': 1.007825,
'He': 4.002603,
'Li': 7.016004,
'Be': 9.012182,
'B': 11.009305,
'C': 12.000000,
'N': 14.003074,
'O': 15.994915,
'F': 18.998403,
'Ne': 19.992440,
'Na': 22.989770,
'Mg': 23.985042,
'Al': 26.981538,
'Si': 27.976927,
'P': 30.973762,
'S': 31.972071,
'Cl': 34.968853,
'Ar': 39.962383,
'K': 38.963707,
'Ca': 39.962591,
'Sc': 44.955910,
'Ti': 47.947947,
'V': 50.943964,
'Cr': 51.940512,
'Mn': 54.938050,
'Fe': 55.934942,
'Co': 58.933200,
'Ni': 57.935348,
'Cu': 62.929601,
'Zn': 63.929147,
'Ga': 68.925581,
'Ge': 73.921178,
'As': 74.921596,
'Se': 79.916522,
'Br': 78.918338,
'Kr': 83.911507,
'Rb': 84.911789,
'Sr': 87.905614,
'Y': 88.905848,
'Zr': 89.904704,
'Nb': 92.906378,
'Mo': 97.905408,
'Tc': 98.907216,
'Ru': 101.904350,
'Rh': 102.905504,
'Pd': 105.903483,
'Ag': 106.905093,
'Cd': 113.903358,
'In': 114.903878,
'Sn': 119.902197,
'Sb': 120.903818,
'Te': 129.906223,
'I': 126.904468,
'Xe': 131.904154,
'Cs': 132.905447,
'Ba': 137.905241,
'La': 138.906348,
'Ce': 139.905435,
'Pr': 140.907648,
'Nd': 141.907719,
'Pm': 144.912744,
'Sm': 151.919729,
'Eu': 152.921227,
'Gd': 157.924101,
'Tb': 158.925343,
'Dy': 163.929171,
'Ho': 164.930319,
'Er': 165.930290,
'Tm': 168.934211,
'Yb': 173.938858,
'Lu': 174.940768,
'Hf': 179.946549,
'Ta': 180.947996,
'W': 183.950933,
'Re': 186.955751,
'Os': 191.961479,
'Ir': 192.962924,
'Pt': 194.964774,
'Au': 196.966552,
'Hg': 201.970626,
'Tl': 204.974412,
'Pb': 207.976636,
'Bi': 208.980383,
'Po': 208.982416,
'At': 209.987131,
'Rn': 222.017570,
'Fr': 223.019731,
'Ra': 226.025403,
'Ac': 227.027747,
'Th': 232.038050,
'Pa': 231.035879,
'U': 238.050783,
'Np': 237.048167,
'Pu': 244.064198,
'Am': 243.061373,
'Cm': 247.070347,
'Bk': 247.070299,
'Cf': 251.079580,
'Es': 252.082972,
'Fm': 257.095099,
'Md': 258.098425,
'No': 259.101024,
'Lr': 262.109692,
'Rf': 267.,
'Db': 268.,
'Sg': 269.,
'Bh': 270.,
'Hs': 270.,
'Mt': 278.,
'Ds': 281.,
'Rg': 282.,
'Cn': 285.,
'Nh': 286.,
'Fl': 289.,
'Mc': 290.,
'Lv': 293.,
'Ts': 294.,
'Og': 294.
}

def get_atom_symbol(atomic_number):
    periodic_table = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]

    if 1 <= atomic_number <= len(periodic_table):
        return periodic_table[atomic_number - 1]
    else:
        return "Invalid atomic number"

def get_atomic_number(atom_symbol):
    periodic_table = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
        "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
        "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
        "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
        "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
        "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
        "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
        "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
        "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
        "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
    }
    return periodic_table.get(atom_symbol, "Invalid atom symbol")
