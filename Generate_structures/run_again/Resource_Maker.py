import os
import pickle

from pymatgen.core import Species, Composition
from pymatgen.core.structure import Structure

import random
from random import sample

from collections import defaultdict

import numpy as np
from numpy.linalg import norm
import scipy as sp

R = sp.spatial.transform.Rotation

Li = Species.from_str("Li+")
Mn3 = Species.from_str("Mn3+")
Mn4 = Species.from_str("Mn4+")
O2 = Species.from_str("O2-")

nn1_dist = 2.97
nn2_dist = 4.2
nn3_dist = 5.14
nn4_dist = 5.94
    
nn_dist = 1.81865
    
def struct_indicies(structure):

    tet_oct_ind = defaultdict(list)
    
    for i,species in enumerate(structure):
        if len(species.species)==3:
            tet_oct_ind['tet'].append(i)
        elif len(species.species)==5:
            tet_oct_ind['oct'].append(i)
        else:
            tet_oct_ind['O2'].append(i)

    return tet_oct_ind

def Nearest_Neighbor_Calculator(structure, tet_oct_ind):

    counter = 0
    
    nn_to_tet = defaultdict(list)
    nn_to_oct = defaultdict(list)
    nns = defaultdict(list)

    for tet_ind in tet_oct_ind['tet']:
        for oct_ind in tet_oct_ind['oct']:
            if abs(structure[tet_ind].distance(structure[oct_ind]) - nn_dist) <= 0.2:
                nn_to_tet[tet_ind].append(oct_ind)
                nn_to_oct[oct_ind].append(tet_ind)
                nns[oct_ind].append(tet_ind)
                nns[tet_ind].append(oct_ind)

    return nn_to_tet, nn_to_oct, nns

def Oct_Oct_Neighbor_Types(tet_oct_ind, structure):

    neighbor_octs = defaultdict(list)

    for oct_ind1 in tet_oct_ind['oct']:
        neighbor_octs[oct_ind1] = {"nn1":[],"nn2":[],"nn3":[],"nn4":[]}
        for oct_ind2 in tet_oct_ind['oct']:
            if oct_ind2!=oct_ind1:
                dist = structure[oct_ind1].distance(structure[oct_ind2])
                if (dist < 3):
                    neighbor_octs[oct_ind1]["nn1"].append(oct_ind2)
                elif (dist > 4) and ( dist < 5):
                    neighbor_octs[oct_ind1]["nn2"].append(oct_ind2)
                elif (dist > 5) and ( dist < 5.5):
                    neighbor_octs[oct_ind1]["nn3"].append(oct_ind2)
                elif (dist > 5.5) and ( dist < 6):
                    neighbor_octs[oct_ind1]["nn4"].append(oct_ind2)
                    
    return neighbor_octs  

def Check_Validity(OO_NNs):
    counter = 0

    for o in list(OO_NNs.keys()):
        if len(OO_NNs[o]['nn1'])!=12:
            counter+=1
            print(OO_NNs[o]['nn1'])
            print(len(OO_NNs[o]['nn1']))

    if counter==0:
        print("The Supercell is large enough to distinguish all ordering types")

    else:
        print("The Supercell is NOT large enough to distinguish all ordering types")

def Central_Atom(s, indices):    # Finds octahedral site (16c or 16d) in the structure closest to [0.5,0.5,0.5] fractional coordinates. 

    mid_point = np.array([0.5,0.5,0.5])

    central_atom = -1

    for l in indices['oct']:
        if (len(s[l].species)==5) and (np.sqrt(np.sum((s[l].frac_coords-mid_point)**2))<0.01):
            central_atom = l
            break

    shortest_distance = 10**6

    if central_atom == -1:
        for l in indices['oct']:
            if (len(s[l].species)==5):
                dist= np.sqrt(np.sum((s[l].frac_coords-mid_point)**2))
                if dist<shortest_distance:
                    shortest_distance = dist
                    central_atom=l
                    
    return central_atom

def Central_Atom_and_Neighbors(central_atom, OO_NNs):  
    
    # Central octahedra and Neighboring Octaherdra needed for transforming between rotational Spinel variants. Returns a list of lists all containing the central atom because the rotational axes allways conatain the central atom.      
    a = OO_NNs[central_atom]['nn1']

    base_site_order =[]

    for i in a:
        b = np.intersect1d(OO_NNs[central_atom]['nn1'],OO_NNs[i]['nn1'])
        for j in b:
            fourth = [k for k in OO_NNs[j]['nn1'] if (k in b) and (k!=i)][0]
            first_tetrahedral = [central_atom, i, j, fourth]
            break
        else:
            continue
        break

    base_site_order.append(first_tetrahedral)

    for corner in first_tetrahedral[1:4]:
        b = np.intersect1d(OO_NNs[central_atom]['nn1'],OO_NNs[corner]['nn1'])
        l = [x for x in b if x not in first_tetrahedral[1:4]]
        tetrahedral = [central_atom, corner]
        for l1 in l:
            tetrahedral.append(l1)
        base_site_order.append(tetrahedral)

    return base_site_order

def Rotation_Calculator(vec1, vec2):

    transform_axis = np.cross(vec1, vec2)/np.linalg.norm(np.cross(vec1, vec2))
    transform_angle = np.arccos(np.dot(vec1, vec2)/(norm(vec1)*norm(vec2)))
    rotation_vector = transform_angle * transform_axis
    rotation = R.from_rotvec(rotation_vector)
    
    return rotation 

def File_Write(filename, obj):
    
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)   

def Ordering_Occupancies(central_atom, base_site_order, s, tet_oct_ind):
    
    #Returns 4 lists of sites in "Occupied". These lists correspond to the sites occupied by TM (or 16d sites) in 4 rotational variants. Translational variants can be derived from these later seperately.
    
    #Supercell Size
    n_a = n_b = n_c = 30
    site = 121935
    base_site = central_atom

    site_order = [site, 67035, 94035, 148035]

    Occupied = defaultdict(list)

    for k in range(len(base_site_order)):
        
        print(f"Order Style {k} \n\n")

        s_spinel_poscar = Structure.from_file('Spinel_POSCAR')

        s_spinel_poscar.make_supercell([[n_a,0,0],[0,n_b,0],[0,0, n_c]])

        Mn_Coordinates = defaultdict(list)

        for i in range(len(s_spinel_poscar)):
            if str(s_spinel_poscar[i].species)=="Mn1":
                Mn_Coordinates[i].append(s_spinel_poscar[i].coords)

        Mn_sites = list(Mn_Coordinates.keys())

        print("Performing the First Transformation of Shifting the origin")
        
        transform_vec = Mn_Coordinates[site][0]-s[base_site].coords

        for mn in Mn_sites:
            Mn_Coordinates[mn][0] -= transform_vec                           #First Transformation of Shifting the origin.

        print("Performing the Second Transformation: Rotation")
            
        vec1 = Mn_Coordinates[site_order[1]][0]-Mn_Coordinates[site][0]
        vec2 = s[base_site_order[k][1]].coords-Mn_Coordinates[site][0]

        for mn in Mn_sites:                                          #Second Transformation: Rotating for second atom agreement.
            rotation = Rotation_Calculator(vec1, vec2)
            Mn_Coordinates[mn][0] = Mn_Coordinates[site][0]+rotation.apply(Mn_Coordinates[mn][0]-Mn_Coordinates[site][0])  

        #Setting up the third transformation
        
        print("Setting up the Third Transformation: Rotation")

        mid_point = list( (np.array(Mn_Coordinates[site_order[1]][0]) + np.array(Mn_Coordinates[site][0]))/2 )
        vec1 = Mn_Coordinates[site_order[2]][0]-mid_point
        vec2 = s[base_site_order[k][2]].coords-mid_point
        rotation = Rotation_Calculator(vec1, vec2)
        
        print("Checking if the Third Transformation is correct. If not, setting up the right one.")
        
        #Check if the third transformation is correct. If not, set up the right one.
        check_coordinates = mid_point+rotation.apply(Mn_Coordinates[site_order[3]][0]-mid_point)                 #Third Transformation: Rotating for third atom agreement.
        if list(np.round(check_coordinates,2))!=list(np.round(s[base_site_order[k][3]].coords,2)):
            vec1 = Mn_Coordinates[site_order[3]][0]-mid_point
            rotation = Rotation_Calculator(vec1, vec2)               
            
        print("Performing the Third Transformation")
            
        for mn in Mn_sites:
            Mn_Coordinates[mn][0] = mid_point+rotation.apply(Mn_Coordinates[mn][0]-mid_point)                 #Third Transformation: Rotating for third atom agreement.

        counter = 0
        
        for octa in tet_oct_ind['oct']:
            for mn in Mn_sites:
                check_distance=np.sqrt(np.sum( (s[octa].coords-Mn_Coordinates[mn][0])**2))
                if check_distance<0.2:
                    Occupied[k].append(octa)
                    counter += 1
            
        if (counter!=(len(s)/8)):   
            composition_deviation = ((len(s)/8)-counter)/(len(s)/8)
            raise AssertionError(f"16d sites are not equal to half of all the octahedral sites!!!!!!! They deviate from their intended number by a fraction {composition_deviation}. For large deviations this is likely because the transformation (probably the rotational transformation) did not take place properly due to the choice of rounding off number. For small deviations (< 5-10 per cent) the supercell size may not be compatible with spinel?")
        
    return Occupied

with open('Transformation_Vectors.pickle', 'rb') as handle:
    Transformation_Vectors = pickle.load(handle)              #Z-axis of the vectors need to be changed to change interfacial distance 
    
with open('base_structure.pickle', 'rb') as handle:
    s = pickle.load(handle)

Interfaces = ['100','110','111','210','211','221']

parent_directory = os.getcwd()

for interface in Interfaces:
    
    Resources = {}
    
    s_base = s.copy()
    s_base.make_supercell(Transformation_Vectors[interface])
    print("Built a super cell")
    
    Resources['initial_structure'] = s_base
    Resources['indices'] = struct_indicies(s_base)
    nn_to_tet, nn_to_oct, Resources['Neighbors'] = Nearest_Neighbor_Calculator(s_base, Resources['indices'])
    Resources['Oct_Neighbors'] = Oct_Oct_Neighbor_Types(Resources['indices'], s_base)

    Check_Validity(Resources['Oct_Neighbors'])
    central_atom = Central_Atom(s_base, Resources['indices'])
    base_site_order = Central_Atom_and_Neighbors(central_atom, Resources['Oct_Neighbors'])

    Resources['Spinel_Orientation_Occupancies'] = Ordering_Occupancies(central_atom, base_site_order, s_base, Resources['indices'])
    
    interface_dir = os.path.join(parent_directory,interface)
    filename = os.path.join(interface_dir, 'Resources.pickle')
    File_Write(filename, Resources)