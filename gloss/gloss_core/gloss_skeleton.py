import numpy as np
import numpy.linalg as npl
import pickle as pkl
import scipy
from scipy import io
import os
import cv2
import gloss.gloss_core.gloss_basic as ba
import pdb
import time
import scipy.sparse as sp
from gloss.gloss_core.gloss_base import Gloss

class GlossSkeleton(Gloss):
    """Global Local Object Shape Space for Animals"""

    def __init__(
        self, end_template_filename='_template',
            end_model_data_filename='_model_data', animal_name='lioness',
            model_dir='../../models/', data_dir='../../training/'):

        self.gender = ''
        self.animal_name = animal_name
        self.template_filename = animal_name + end_template_filename
        self.model_data_filename = animal_name + end_model_data_filename
        self.pose_filename = animal_name + '_pose'
        self.model_dir = model_dir
        self.data_dir = data_dir

        self.colors = ['turquoise3' , 'LightSalmon', 'firebrick', 'LavenderBlush3', 'CornflowerBlue',\
            'DarkOliveGreen3', 'maroon3', 'chartreuse3', 'coral3' ,'thistle3', 'gold3', 'LightPink3',\
            'orchid3', 'OrangeRed3', 'PaleTurquoise3', 'aquamarine3','RoyalBlue3', 'gold3', 'LightSkyBlue3', \
            'SpringGreen3', 'sienna3', 'GhostWhite', 'thistle3', 'RoyalBlue3',  \
            'LavenderBlush3', 'MediumSlateBlue', 'pink','MistyRose','slate grey','wheat3', 'MediumAquamarine','OliveDrab',
            'GreenYellow', 'PaleGreen', 'light coral','red'  ]

    def load_model(self, name):
        filename = name + ".pkl"
        load_file = os.path.join(self.model_dir, filename)
        with open(load_file, 'rb') as f:
            data = pkl.load(f, encoding='latin1')

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
        self.nBshape = data['nBshape']
        self.nMaxShapeBasis = data['nMaxShapeBasis']
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
        self.shapePCA = data['shapePCA']
        self.interfacePointsFromTo = data['interfacePointsFromTo']
        self.nShapeBasis = np.zeros(self.nParts, dtype=np.int)
        self.Zs = [None]*self.nParts
        for i in self.partSet:
            self.nShapeBasis[i] = len(self.shapePCA[i]['sigma'])
            self.Zs[i] = np.zeros((self.nShapeBasis[i]))
        self.t = np.zeros((self.nParts, 3))
        self.r_abs = np.zeros((self.nParts, 3))
        self.template_points = data['template_points']
        self.tri = data['tri']
        self.seg = data['seg']
        return

