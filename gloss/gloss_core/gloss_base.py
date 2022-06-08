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
from psbody.mesh.meshviewer import MeshViewers
from psbody.mesh import Mesh


def create_cage(vs):
    from itertools import product
    from psbody.mesh.sphere import Sphere
    return([Sphere(np.asarray(corner), 1e-10).to_mesh()
            for corner in product(*zip(vs.min(axis=0), vs.max(axis=0)))])


class Gloss(object):
    """Global Local Object Shape Space"""

    partPoints = []
    part2bodyPoints = []
    body2partPoints = []
    partFaces = []
    interfaceFacesFromTo = []
    interfacePointsFromTo = []
    posePCA = []
    poseDefModel = []
    poseDefModelNeighbors = []
    poseDefModelA2B = []
    partSet = []
    pairs = []
    parent = []
    child = []
    parts = []
    names = []
    parts2Dpoints = []

    def __init__(self):
        # The parts should be orderd from root to leaves

        self.template_filename = ''
        self.pose_filename = 'scmp_ho'
        self.data_dir = '../../training/'
        self.model_dir = '../../models/'
        self.shapePCA = []
        self.posePCA = []
        self.nMaxPoseBasis = 60
        self.nMaxShapeBasis = 60
        self.template_r_abs = 0
        self.template_t = 0
        self.template_skel = 0
        self.fixedShape = False  # To test
        self.colors = [
            'turquoise3', 'LightSalmon', 'firebrick',
            'red', 'wheat3',
            'DarkOliveGreen3', 'OrangeRed3', 'maroon3',
            'chartreuse3', 'coral3', 'red', 'gold3',
            'LightPink3', 'orchid3', 'PaleTurquoise3',
            'aquamarine3', 'RoyalBlue3', 'gold3', 'LightSkyBlue3',
            'SpringGreen3', 'sienna3', 'GhostWhite', 'thistle3',
            'RoyalBlue3', 'LavenderBlush3', 'gold3','pink',
            'MistyRose','slate grey','CornflowerBlue', 'MediumAquamarine',
            'OliveDrab', 'GreenYellow', 'PaleGreen', 'light coral', 'red', 'gray', 'green']

    def get_part_mesh(self, i, Zp, Zs):
        P = ba.get_part_mesh(self, i, Zp, Zs)
        return P

    def abs_to_rel(self, r_abs):
        r_rel = r_abs.copy()
        for part in self.partSet:
            parent = self.parent[part]
            if parent >= 0:
                R0_parent, _ = cv2.Rodrigues(r_abs[parent,:])
                R0_part, _ = cv2.Rodrigues(r_abs[part,:])
                R = R0_parent.T.dot(R0_part)
                r, _ = cv2.Rodrigues(R)
                r_rel[part,:] = r.T
        return r_rel

    def rel_to_abs(self, r_rel):
        r_abs = r_rel.copy()
        for part in self.partSet:
            parent = self.parent[part]
            if parent >= 0:
                R_parent, _ = cv2.Rodrigues(r_abs[parent,:])
                R_part, _ = cv2.Rodrigues(r_rel[part,:])
                R = R_parent.dot(R_part)
                r, _ = cv2.Rodrigues(R)
                r_abs[part,:] = r.T
        return r_abs

    def t_abs_to_rel(self, t_abs, r_abs=None):
        t_rel = t_abs.copy()
        for part in self.partSet:
            parent = self.parent[part]
            if parent >= 0:
                if r_abs is None:
                    t_rel[part,:] = t_abs[part,:] - t_abs[parent,:]
                else:
                    # relative translation in t-pose
                    R, _ = cv2.Rodrigues(r_abs[parent,:])
                    t_rel[part,:] = R.T.dot(t_abs[part,:] - t_abs[parent,:])
        return t_rel


    def show_model(
                  self, PCA, add_cage=False, forPart=None, nPC=7,
                  nSigma=3, in_color=None, stop=False):
        mvs = MeshViewers(shape=(nPC, 4))

        if forPart is not None:
            partSet = [forPart]
        else:
            partSet = self.partSet

        for part in partSet:
            if in_color is None:
                color = self.colors[part]
            else:
                color = in_color
            minX = np.inf
            maxX = -np.inf
            minY = np.inf
            maxY = -np.inf
            minZ = np.inf
            maxZ = -np.inf
            M = PCA[part]['M']+PCA[part]['T']
            B = PCA[part]['B']
            for h in range(0, nPC):
                sigma = PCA[part]['sigma']
                if add_cage:
                    for im, m in enumerate([-nSigma, 0, nSigma]):
                        A = M + B[:, h]*m*sigma[h]
                        X = A[0::3]
                        if minX > min(X):
                            minX = min(X)
                        if maxX < max(X):
                            maxX = max(X)
                        Y = A[1::3]
                        if minY > min(Y):
                            minY = min(Y)
                        if maxY < max(Y):
                            maxY = max(Y)
                        Z = A[2::3]
                        if minZ > min(Z):
                            minZ = min(Z)
                        if maxZ < max(Z):
                            maxZ = max(Z)
                    v = np.array([[minX, minY, minZ], [maxX, maxY, maxZ]])
                    cage = create_cage(v)
                    ms = [cage[0]]
                    for j in range(1, len(cage)):
                        ms.append(cage[j])
                A = M
                X = A[0::3]
                Y = A[1::3]
                Z = A[2::3]
                v = np.zeros((len(X), 3))
                v[:, 0] = X
                v[:, 1] = Y
                v[:, 2] = Z
                mgm = Mesh(v=v, f=self.partFaces[part])
                mgm.set_vertex_colors('red')
                mg = [None]*3
                for im, m in enumerate([-nSigma, 0, nSigma]):
                    A = M + B[:, h]*m*sigma[h]
                    X = A[0::3]
                    Y = A[1::3]
                    Z = A[2::3]
                    v = np.zeros((len(X), 3))
                    v[:, 0] = X
                    v[:, 1] = Y
                    v[:, 2] = Z
                    mg[im] = Mesh(v=v, f=self.partFaces[part])
                    mg[im].set_vertex_colors(color)
                    if add_cage:
                        mvs[nPC-1-h][im].set_dynamic_meshes(ms+[mg[im]])
                    else:
                        mvs[nPC-1-h][im].set_dynamic_meshes([mg[im]])
                mg[0].set_vertex_colors('red')
                mg[1].set_vertex_colors('green')
                mg[2].set_vertex_colors('blue')
                mvs[nPC-1-h][3].set_static_meshes(mg)
            if stop:
                pdb.set_trace()

    def save_model(self, name):
        return

    def learn_model(self, name):
        return

    def load_model(self, name):
        filename = name + ".pkl"
        load_file = os.path.join(self.model_dir, filename)
        with open(load_file) as f:
            data = pkl.load(f)

        self.partFaces = data['partFaces']
        self.part2bodyPoints = data['part2bodyPoints']
        self.body2partPoints = data['body2partPoints']
        self.partPoints = data['partPoints']
        self.posePCA = data['posePCA']
        self.poseDefModelA2B = data['poseDefModelA2B']
        self.shapePCA = data['shapePCA']
        self.interfacePointsFromTo = data['interfacePointsFromTo']
        try:
            self.leftPartSet = data['leftPartSet']
            self.rightPartSet = data['rightPartSet']
        except:
            print('left and right part sets are not defined')
            pass

        try:
            self.mesh_interfacePointsFromTo = data['mesh_interfacePointsFromTo']
        except:
            print('mesh_interfacePointsFromTo is not defined')
            pass
        try:
            self.interfacePointsFlex = data['interfacePointsFlex']
        except:
            pass
        self.interfaceBones = data['interfaceBones']
        for i in self.partSet:
            self.nPoseBasis[i] = len(self.posePCA[i]['sigma'])
            self.nShapeBasis[i] = len(self.shapePCA[i]['sigma'])
            self.Zp[i] = np.zeros((self.nPoseBasis[i]))
            self.Zs[i] = np.zeros((self.nShapeBasis[i]))
        self.t = np.zeros((self.nParts, 3))
        self.r_abs = np.zeros((self.nParts, 3))
        try:
            self.template_t = data['template_t']
            self.template_r_abs = data['template_r_abs']
        except:
            print('model without template pose')

        filename = self.template_filename + '.pkl'
        print('loading ' + filename)
        data = self.load_sample(filename)
        self.template_points = data['points']
        self.tri = data['tri']
        self.seg = data['seg']
        return
