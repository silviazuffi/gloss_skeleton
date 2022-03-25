"""
  
    Author: Silvia Zuffi
    silvia@mi.imati.cnr.it

"""
import numpy as np
import numpy.random as npr
import pickle as pkl
import pdb
import cv2
import gloss.gloss_basic as ba
import time
import chumpy as ch
from gloss.gsp import SpVerts


def random_rot():
    s = npr.rand()
    s1 = np.sqrt(1-s)
    s2 = np.sqrt(s)
    theta_1 = 2.0*np.pi*npr.rand()
    theta_2 = 2.0*np.pi*npr.rand()
    w = np.cos(theta_2)*s2
    x = np.sin(theta_1)*s1
    y = np.cos(theta_1)*s1
    z = np.sin(theta_2)*s2

    a = 2.0 * np.arccos(w)
    r = np.zeros(3)
    d = np.sqrt(1-w**2)
    if d > 0.0:
        r[0] = a*x / d
        r[1] = a*y / d
        r[2] = a*z / d
    return r


def random_axis(axis):

    u = -1.0 + 2.0*npr.rand()
    theta = 2.0*np.pi * npr.rand()
    axis[0] = np.sqrt(1 - u**2) * np.cos(theta)
    axis[1] = np.sqrt(1 - u**2) * np.sin(theta)
    axis[2] = u


def get_sample(this, nB_pose=4, nB_shape=4, fixedShape=True, add_global_rotation=True,
               init_torso_rotation=np.zeros((3)), init_rotation_axis=None,
               multi_shape_PCA=False, samplePart=None, idx=None, inputZs=None, has_pose_space=True):
    """
    Returns the set of points of a model sample obtained with the specified number of basis.
    The number of basis can be a vector of dimension number of parts, or a single value for
    each part.
    Also returns the model parameters
    """

    Zp = [None] * (np.max(this.partSet)+1)
    if nB_pose is not None:
        if np.size(nB_pose) == 1:
            nB_p = nB_pose*np.ones((np.max(this.partSet)+1), dtype=np.int)
        else:
            nB_p = nB_pose
        fixedPose = False
    else:
        fixedPose = True

    # Generate the deformations for each part
    def generate_children_deformations(node):
        if node == this.root:
            Zp[node] = npr.normal(np.zeros((nB_p[node])), this.posePCA[node]['sigma'][:nB_p[node]])
        cInd = np.arange(0, nB_p[node])
        x = Zp[node]
        for b in this.children[node]:
            if b >= 0:
                rInd = np.arange(this.nPoseBasis[node], this.nPoseBasis[node]+nB_p[b])
                try:
                    mu = this.poseDefModelA2B[node][b]['mu']
                    C = this.poseDefModelA2B[node][b]['C']
                    mu_ab, C_ab = ba.compute_conditioned_gaussian(this, rInd, cInd, mu, C, x)
                    Zp[b] = mu_ab
                except:
                    Zp[b] = npr.normal(np.zeros((nB_p[b])), this.posePCA[b]['sigma'][:nB_p[b]])
                generate_children_deformations(b)

    if fixedPose:
        for b in this.partSet:
            #Zp[b] = np.zeros((this.nPoseBasis[b]))
            Zp[b] = np.zeros((10))
    else:
        generate_children_deformations(this.root)

    if fixedShape:
        zs = np.zeros((this.nShapeBasis[this.root]))
        Zs = [zs for i in this.partSet]
    else:
        if multi_shape_PCA:
            Zs = [npr.normal(np.zeros(nB_shape), this.shapePCA[i]['sigma'][:nB_shape]) for i in this.partSet]
        else:
            if samplePart is None:
                samplePart = this.root
            zs = npr.normal(np.zeros(nB_shape), this.shapePCA[samplePart]['sigma'][:nB_shape])
            Zs = [zs for i in this.partSet]
    if idx is not None:
        if idx >= 0: 
            for i in this.partSet:
                Zs[i][:] = 0.0
                Zs[i][idx] = 3.0*this.shapePCA[i]['sigma'][idx]
        else:
            for i in this.partSet:
                Zs[i][:] = 0.0
                Zs[i][-idx] = -3.0*this.shapePCA[i]['sigma'][-idx]

    if inputZs is not None:
        Zs = inputZs

    # Generate the mesh
    Pw = [None]*(np.max(this.partSet)+1)
    r_abs = np.zeros(((np.max(this.partSet)+1), 3))

    t = np.zeros(((np.max(this.partSet)+1), 3))

    def compute_stitching(node):
        if node == this.root:
            # Add a global rotation to the torso
            if init_rotation_axis is not None:
                alpha = npr.rand(1)
                r_abs[node, init_rotation_axis] = -np.pi + alpha * 2.0*np.pi
            elif add_global_rotation:
                r_abs[node, :] = random_rot()
            else:
                r_abs[node, :] = 0.0
            R1, _ = cv2.Rodrigues(r_abs[node, :])
            R2, _ = cv2.Rodrigues(init_torso_rotation)
            R = np.matrix(R2)*np.matrix(R1)
            r, _ = cv2.Rodrigues(R)
            r_abs[node, :] = r[:, 0]
            T = np.zeros((3))
            P = ba.get_part_mesh(this, node, Zp[node], Zs[node], has_pose_space=has_pose_space)
            Pw[node] = ba.object_to_world(P, R, T)
            r, _ = cv2.Rodrigues(R)
            r_abs[node, :] = r.flatten()
            t[node, :] = T.flatten()

        for nb in this.children[this.names[node]]:
            b = this.parts[nb]
            if b >= 0:
                P = ba.get_part_mesh(this, b, Zp[b], Zs[b], has_pose_space=has_pose_space)
                if fixedPose:
                    R = np.eye(3)
                else:
                    R = None
                R, T = ba.align_to_parent(this, b, node, P, Pw[node], R)
                Pw[b] = ba.object_to_world(P, R, T)
                r, _ = cv2.Rodrigues(R)
                r_abs[b, :] = r.flatten()
                t[b, :] = T.flatten()
                compute_stitching(b)

    compute_stitching(this.root)

    return Pw, r_abs, t, Zp, Zs

