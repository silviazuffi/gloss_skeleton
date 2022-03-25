"""

    Author: Silvia Zuffi
    silvia@mi.imati.cnr.it

"""



import os
import pickle as pkl
import numpy as np

class StitchedSkeleton():
    """Global Local Object Shape Space for Animals"""

    def __init__(self):

        self.colors = ['turquoise3' , 'LightSalmon', 'firebrick', 'LavenderBlush3', 'CornflowerBlue',\
            'DarkOliveGreen3', 'maroon3', 'chartreuse3', 'coral3' ,'thistle3', 'gold3', 'LightPink3',\
            'orchid3', 'OrangeRed3', 'PaleTurquoise3', 'aquamarine3','RoyalBlue3', 'gold3', 'LightSkyBlue3', \
            'SpringGreen3', 'sienna3', 'GhostWhite', 'thistle3', 'RoyalBlue3',  \
            'LavenderBlush3', 'MediumSlateBlue', 'pink','MistyRose','slate grey','wheat3', 'MediumAquamarine','OliveDrab',
            'GreenYellow', 'PaleGreen', 'light coral','red'  ]


    def load_model(self, model_file):
        with open(model_file, 'rb') as f:
            data = pkl.load(f, encoding='latin1')

        # import ipdb; ipdb.set_trace()
        self.root = data['root']
        self.partSet = data['partSet']
        self.pairs = data['pairs']
        self.nPairs = len(self.pairs)
        
        self.parent = data['parent']
        self.children = data['children']
        self.parts = data['parts']
        self.nParts = np.max(self.partSet)+1
        self.neighbors = data['neighbors']
        self.names = data['names']
        
        self.rmin = data['rmin']
        self.rmax = data['rmax']
        self.template_r_abs = data['template_r_abs']
        self.template_t = data['template_t']
        
        self.tri = data['tri']
        self.seg = data['seg']

        self.partFaces = data['partFaces']
        self.part2bodyPoints = data['part2bodyPoints']
        self.body2partPoints = data['body2partPoints']
        self.partPoints = data['partPoints']
        if 'shape' in data.keys():
            self.shape = data['shape']
        if 'shapePCA' in data.keys():
            self.shapePCA = data['shapePCA']
        self.interfacePointsFromTo = data['interfacePointsFromTo']

        self.t = np.zeros((self.nParts, 3))
        self.r_abs = np.zeros((self.nParts, 3))
        self.template_points = data['template_points']
        self.tri = data['tri']
        self.seg = data['seg']
        return

