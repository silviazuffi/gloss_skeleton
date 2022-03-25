"""

    Author: Silvia Zuffi
    silvia@mi.imati.cnr.it

"""

from chumpy import Ch, depends_on
import chumpy as ch
import numpy as np
import cv2
import pdb
from psbody.mesh.colors import name_to_rgb
from psbody.mesh import Mesh
import pickle as pkl

def shape_idx(part, nB):
    n = np.sum(nB[:part])
    idx = range(n,n+nB[part])
    return idx
def pose_idx(part, nB):
    n = np.sum(nB[:part])
    idx = range(n,n+nB[part])
    return idx

class Rodrigues(Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T

def abs_to_rel(model, r_abs):
    partSet = model.partSet
    parent = model.parent
    r_rel = r_abs[0,:].copy()
    for part in partSet:
        parnt_name = parent[model.names[part]]
        if parnt_name != []:
            parnt = model.parts[parnt_name]
            R0_parent = Rodrigues(r_abs[parnt])
            R0_part = Rodrigues(r_abs[part])
            R = R0_parent.T.dot(R0_part)
            r = Rodrigues(R)
            r_rel = ch.vstack([r_rel, r.T])
    return r_rel


def rel_to_abs(model, r_rel):
    partSet = model.partSet
    parent = model.parent

    r_abs = r_rel[model.root,:].copy()
    for part in partSet:
        parnt_name = parent[model.names[part]]
        if parnt_name != []:
            parnt = model.parts[parnt_name]
            R_part = Rodrigues(r_rel[part])
            try:
                R_parent = Rodrigues(r_abs[parnt,:])
            except:
                R_parent = Rodrigues(r_abs)
            R = R_parent.dot(R_part)
            r = Rodrigues(R)
            r_abs = ch.vstack([r_abs, r.T])
    return r_abs


class SpVerts(Ch):

    dterms = ('t', 'r_abs')
    terms = ('model')
    
    def __init__(self, skel_model, r_abs=None, t=None):
        self.model = skel_model
        self.init_sp()

    def set_r_rel(self, r_rel):
        self.r_abs = rel_to_abs(self.model, r_rel.reshape((-1,3))).ravel()
       
    def get_r_rel(self):
        return abs_to_rel(self.model, self.r_abs.reshape((-1,3)))

    def compute_part_points(self, part, r_abs, t):
        P = self.model.shape[part]['T'] 
        R = Rodrigues(r_abs)
        P = P.reshape(int(len(P)/3),3)
        Pw = P.dot(R.T)+t
        return Pw, P



    def compute_part_points_dr_wrt(self, part, r_abs, t, wrt):
        if wrt is not self.t and wrt is not self.r_abs:
            return

        nPoints = int(self.model.shape[part]['T'].shape[0]/3)
        if wrt is self.t:
            dr = np.zeros((nPoints, 3, 3*self.model.nParts))
            dr[:,0,3*part] = 1.0
            dr[:,1,3*part+1] = 1.0
            dr[:,2,3*part+2] = 1.0
            dr = dr.reshape([nPoints*3,3*self.model.nParts])

        elif wrt is self.r_abs:
            dr = np.zeros((nPoints, 3, 3*self.model.nParts))
            rr = Rodrigues(r_abs) 
            R = rr.r
            J = rr.compute_dr_wrt(rr.rt).T

            _, r = self.compute_part_points(part, r_abs, t)
            i = 3*part
            for l in range(3):
                Jr = J[l,:].reshape((3,3))
                F = np.dot(r,Jr.T)
                dr[:,:,i+l] =  F
            dr = dr.reshape([nPoints*3,3*self.model.nParts])

        return dr

    def compute_r(self):
        r_abs = ch.reshape(self.r_abs, (int(len(self.r_abs)/3),3))
        t = ch.reshape(self.t, (int(len(self.t)/3),3))
        nV = 0
        vert_idx_shift = np.zeros((self.model.nParts), dtype=np.uint32)
        # Compute also the faces and the indexes to define the stitching cost
        for part in self.model.partSet:
            if part == self.model.partSet[0]:
                P  =  self.compute_part_points(part, r_abs[part,:], t[part,:])[0].r
                f = self.model.partFaces[part]
            else:
                Ppart  =  self.compute_part_points(part, r_abs[part,:], t[part,:])[0].r
                P =  ch.vstack([P, Ppart])
                f = np.vstack([f, self.model.partFaces[part]+nV])
            vert_idx_shift[part] = nV
            nV = nV + int(self.model.shape[part]['T'].shape[0]/3)
        return P

    def init_sp(self):
        r_abs = ch.reshape(self.r_abs, (int(len(self.r_abs)/3),3))
        t = ch.reshape(self.t, (int(len(self.t)/3),3))
        nV = 0
        vert_idx_shift = np.zeros((self.model.nParts), dtype=np.uint32)
        # Compute also the faces and the indexes to define the stitching cost
        for part in self.model.partSet:
            if part == self.model.partSet[0]:
                P  =  self.compute_part_points(part, r_abs[part,:], t[part,:])[0].r
                f = self.model.partFaces[part]
            else:
                Ppart  =  self.compute_part_points(part, r_abs[part,:], t[part,:])[0].r
                P =  ch.vstack([P, Ppart])
                f = np.vstack([f, self.model.partFaces[part]+nV])
            vert_idx_shift[part] = nV
            nV = nV + int(self.model.shape[part]['T'].shape[0]/3)
        self.f = f


        use_mesh_pairs = False
        Interfaces = self.model.pairs[:int(len(self.model.pairs)/2)]

        k = 0
        for idx, pairs in enumerate(Interfaces):
            i = pairs[0]
            j = pairs[1]
         
            if use_mesh_pairs: 
                if k == 0:
                    Ii = self.model.mesh_interfacePointsFromTo[i][j][::1] + vert_idx_shift[i]
                    Ij = self.model.mesh_interfacePointsFromTo[j][i][::1] + vert_idx_shift[j]
                    pairIdx = idx*np.ones(len(self.model.mesh_interfacePointsFromTo[j][i][::1]), dtype=np.uint8)
                    k = k+1
                else:
                    Ii = np.hstack([Ii, self.model.mesh_interfacePointsFromTo[i][j][::1] + vert_idx_shift[i]])
                    Ij = np.hstack([Ij, self.model.mesh_interfacePointsFromTo[j][i][::1] + vert_idx_shift[j]])
                    pairIdx = np.hstack([pairIdx, idx*np.ones(len(self.model.mesh_interfacePointsFromTo[j][i][::1]), dtype=np.uint8)])
            else:
                if k == 0:
                    Ii = self.model.interfacePointsFromTo[i][j][::1] + vert_idx_shift[i]
                    Ij = self.model.interfacePointsFromTo[j][i][::1] + vert_idx_shift[j]
                    pairIdx = idx*np.ones(len(self.model.interfacePointsFromTo[j][i][::1]), dtype=np.uint8)
                    k = k+1
                else:
                    Ii = np.hstack([Ii, self.model.interfacePointsFromTo[i][j][::1] + vert_idx_shift[i]])
                    Ij = np.hstack([Ij, self.model.interfacePointsFromTo[j][i][::1] + vert_idx_shift[j]])
                    pairIdx = np.hstack([pairIdx, idx*np.ones(len(self.model.interfacePointsFromTo[j][i][::1]), dtype=np.uint8)])

        self.interfIdx = (Ii, Ij)
        self.pairIdx = pairIdx
        self.vertIdxShift = vert_idx_shift

        return P


    def compute_dr_wrt(self, wrt):
        if wrt is not self.t and wrt is not self.r_abs:
            return
        r_abs = ch.reshape(self.r_abs, (int(len(self.r_abs)/3),3))
        t = ch.reshape(self.t, (int(len(self.t)/3),3))
        for part in self.model.partSet:
            if part == self.model.partSet[0]:
                D = self.compute_part_points_dr_wrt(part, r_abs[part,:], t[part,:], wrt)
            else:
                D = ch.vstack([D, self.compute_part_points_dr_wrt(part, r_abs[part,:], t[part,:], wrt)])

        return D

    def get_colored_mesh(self):
        A = self.to_template_topology(self.r)
        colors = np.zeros((len(self.model.tri),3))
        for part in self.model.partSet:
            pidx = np.where(self.model.seg==part)[0]
            colors[pidx,:] = name_to_rgb[self.model.colors[part]]
        M = Mesh(v=A, f=self.model.tri)
        M.set_face_colors(colors)
        return M, colors

    def get_colored_mesh_disconnected(self):
        colors = np.zeros((self.r.shape))
        N = 0
        for part in self.model.partSet:
            n = int(len(self.model.shape[part]['T'])/3)
            colors[N:n+N,:] = name_to_rgb[self.model.colors[part]]
            N = n+N
        M = Mesh(v=self.r, f=self.f)
        M.set_vertex_colors(colors)
        return M, colors

    def get_vlabels(self):
        labels = np.zeros((self.r.shape[0]), dtype=int)
        N = 0
        for part in self.model.partSet:
            n = int(len(self.model.shape[part]['T'])/3)
            labels[N:n+N] = part
            N = n+N
        return labels

    def to_template_topology(self, A):
        '''
        Convert model topology with duplicated vertices into unique vertices
        '''
        if np.size(A.shape) == 3:
            # A derivative
            assert(A.shape[1]==3)
            H = np.zeros((self.model.template_points.shape[0], 3, A.shape[2]))
            k = 0
            for part in self.model.partSet:
                n = int(self.model.shape[part]['T'].shape[0]/3)
                H[self.model.part2bodyPoints[part],:,:] = A[k:k+n,:,:]
                k = k + n
        elif np.size(A.shape) == 2:
            # An array of vertices
            assert(A.shape[1]==3)
            H = np.zeros((self.model.template_points.shape[0], 3))
            k = 0
            for part in self.model.partSet:
                n = int(self.model.shape[part]['T'].shape[0]/3)
                H[self.model.part2bodyPoints[part],:] = A[k:k+n,:]
                k = k + n
        else:
            print('error')
        return H

    def from_template_topology(self, A):
        """
        """
        if np.size(np.array(A).shape) == 1:
            # A set of indexes to convert from the 
            # template topology to this model topology
            H = np.zeros(np.array(A).shape, dtype=int)
            for idx,i in enumerate(A):
                # Find the face that include the vertex
                f = np.where(self.model.tri==i)[0]
                # Find the parts that include the faces
                part = self.model.seg[f][0]
                j = np.where(self.model.part2bodyPoints[part]==i)[0]
                H[idx] = j + self.vertIdxShift[part]
            return H
        elif (A.shape[1]==3):
            for part in self.model.partSet:
                if part == self.model.partSet[0]:
                    H = A[self.model.part2bodyPoints[part],:] 
                else:
                    H = np.vstack([H, A[self.model.part2bodyPoints[part],:]])
            return H
        else:
            print('error')

 
    def get_interface_points_weights(self):
        W = np.ones(self.interfIdx[0].shape[0])
        Interfaces = self.model.mesh_pairs[:int(len(self.model.mesh_pairs)/2)]
        n = 0
        for pair in Interfaces:
            a = pair[0]
            c = pair[1]
            N = len(self.model.mesh_interfacePointsFromTo[a][c])
            W[n:n+N] = 1.0/float(N)
            n = n+N
        W = W/np.sum(W)

        return W

    def global_X_rot(self, angle):
        RotX = np.array([
            [1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
            [0., np.sin(angle), np.cos(angle)]
            ])
        rx,_ = cv2.Rodrigues(RotX)
        self.r_abs[::3] = rx[0]
        self.r_abs[1::3] = rx[1]
        self.r_abs[2::3] = rx[2]
        T = self.t.r.reshape((-1,3))
        Trot = T.dot(RotX.T)
        self.t[:] = Trot.flatten()
        return
    def global_Y_rot(self, angle):
        RotY = np.array([
            [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
            [-np.sin(angle), 0., np.cos(angle)]
            ])
        ry,_ = cv2.Rodrigues(RotY)
        self.r_abs[::3] = ry[0]
        self.r_abs[1::3] = ry[1]
        self.r_abs[2::3] = ry[2]
        T = self.t.r.reshape((-1,3))
        Trot = T.dot(RotY.T)
        self.t[:] = Trot.flatten()
        return
    def global_Z_rot(self, angle):
        RotZ = np.array([
            [np.cos(angle), -np.sin(angle), 0.], 
            [np.sin(angle), np.cos(angle), 0.],
            [0., 0., 1.],
            ])
        rz,_ = cv2.Rodrigues(RotZ)
        self.r_abs[::3] = rz[0]
        self.r_abs[1::3] = rz[1]
        self.r_abs[2::3] = rz[2]
        T = self.t.r.reshape((-1,3))
        Trot = T.dot(RotZ.T)
        self.t[:] = Trot.flatten()
        return





