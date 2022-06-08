"""

    Author: Silvia Zuffi

    Functions for basic computations and visualization with the body model.

"""

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import pickle as pkl
import pdb
from psbody.mesh.meshviewer import MeshViewer
from psbody.mesh.mesh import Mesh
import pylab
from pylab import plt
import cv2
from scipy.spatial.distance import cdist
from scipy import signal
import scipy
import scipy.ndimage
import time
from chumpy.utils import row, col
from opendr.simple import *


def get_part_mesh(this, part, Zp=None, Zs=None):
    """
    Returns the points P(nx3) of the part in local frame for PCA pose and
    shape coefficients Zp and Zs respectively
    """

    # Compute shape-varying template
    if Zs is None:
        # Do not use the shape model, so we do not add the mean
        Tp = this.shapePCA[part]['T']
    else:
        nB_s = len(Zs)
        B = this.shapePCA[part]['B']
        Tp = this.shapePCA[part]['T']+this.shapePCA[part]['M']+ \
            np.array(B[:, 0:nB_s]*np.matrix(Zs).T)[:, 0]

    # Add mean pose deformations
    if Zp is not None:
        M = this.posePCA[part]['M']+Tp

    # Add pose deformations
    if Zp is not None:
        B = this.posePCA[part]['B']

    if Zp is None:
        A = Tp
    else:
        nB_p = len(Zp)
        A = M + np.array(B[:, 0:nB_p]*np.matrix(Zp).T)[:, 0]

    P = np.zeros((int(len(A)/3), 3))
    P[:, 0] = A[0::3]
    P[:, 1] = A[1::3]
    P[:, 2] = A[2::3]

    return P


def object_to_world(P, R, T):
    """
    Computes the points in global frame with the transformation R,T
    Input points are P(nx3), R(3x3) and T is a 3-dim vector
    """

    Pw = P.dot(R.T)+T
    return Pw


def world_to_object(Pw, R, T):
    # Converts points in global frame in the frame specified by R,T
    P = (Pw-T).dot(R)
    return P


def get_rotation_from_3D_points(P, Pw):
    R, _ = rigid_transform_3D(P, Pw)
    return R


def align_to_parent_detached(this, part, parent, P, Pparent):
    """
    Computes the transformation to align a part expressed in local frame
    to its parent expressed in global frame. 
    """

    cl = this.interfacePointsFromTo[part][parent]
    clp = this.interfacePointsFromTo[parent][part]

    T = np.mean(Pparent[clp, :],0) - np.mean(P[cl, :],0)
    return None, T

def align_to_parent(this, part, parent, P, Pparent, R0):
    """
    Computes the transformation to align a part expressed in local frame
    to its parent expressed in global frame. 
    """

    cl = this.interfacePointsFromTo[part][parent]
    clp = this.interfacePointsFromTo[parent][part]

    if R0 is None:
        R, T = rigid_transform_3D(P[cl, :], Pparent[clp, :])
    else:
        R = R0
        T = Pparent[clp, :] - P[cl, :]*np.matrix(R).T
        T = np.mean(T, 0)
    return R, T


def rigid_transform_3D(A, B):
    # Computes the rigid transformation to align point sets A and B

    c_A = np.mean(A, 0)
    c_B = np.mean(B, 0)
    H = ((A - c_A).T).dot(B - c_B)
    U, S, V = npl.svd(H)
    V = np.matrix(V).T
    W = np.eye(U.shape[0])
    W[-1, -1] = npl.det(V.dot(U.T))
    R = V.dot(W).dot(U.T)
    if npl.det(R) < 0:
        # reflection
        V[:, 2] = -1.0*V[:, 2]
        R = V*np.matrix(U).T
    r, _ = cv2.Rodrigues(R)
    T = -R.dot(c_A.T)+c_B
    return R, T


def get_part_mesh_from_parameters(this, part, t, r_abs, Zs=None, Zp=None):

    P = get_part_mesh(this, part, Zp, Zs)
    R, _ = cv2.Rodrigues(r_abs)
    Pw = object_to_world(P, R, t)
    return Pw


def get_colored_mesh(this):

    P = get_sbm_points_per_part(this)
    me = [Mesh(v=P[part], f=this.partFaces[part]).set_vertex_colors(this.colors[part]) for part in this.partSet]
    return me



def gloss_to_smpl_mesh(this, Pw=None):
    """
    Transforms a set of part meshes in a smpl mesh.
    Note: it does not compute the average vertex at the
    interface points!!
    """
    f = this.tri
    v = this.template_points.copy()

    if Pw is None:
        Pw = [None]*this.nParts
    for part in this.partSet:
        if Pw[part] is None:
            P = get_part_mesh(this, part, this.Zp[part], this.Zs[part])
            R, _ = cv2.Rodrigues(this.r_abs[part, :])
            T = this.t[part, :]
            Pw[part] = object_to_world(P, R, T)
        pidx = this.part2bodyPoints[part]
        v[pidx, :] = Pw[part]

    return v, f

def get_sbm_points_per_part(this):
    Pw = [None]*(np.max(this.partSet)+1)
    for part in this.partSet:
        parent = this.parent[part]
        P = get_part_mesh(this, part, this.Zp[part], this.Zs[part])
        R, J = cv2.Rodrigues(this.r_abs[part, :])
        T = this.t[part, :]
        Pw[part] = object_to_world(P, R, T)
    return Pw

def get_skeleton(this, v=None):
    """
    Defines a skeleton by computing the mean of the interface points.
    Returns an array of size (nParts,2,3)
    The first point is the joint with the parent, the second point
    is the part center

    TODO: It does not work for the leaves.

    """
    if v is None:
        Pw = get_sbm_points_per_part(this)
    else:

        Pw = [None]*this.nParts
        for i in this.partSet:
            Pw[i] = v[this.part2bodyPoints[i],:]
        
    bones = np.zeros((this.nParts, 2, 3))
    try:
        t = this.t
    except:
        this.t = 0
        t = this.t
    for i in this.partSet:
        if i == this.root:
            p = this.children[i][0]
            b1 = np.mean(Pw[i][this.interfacePointsFromTo[i][p], :], axis=0)
            b2 = np.mean(Pw[p][this.interfacePointsFromTo[p][i], :], axis=0)
            bones[i][0, :] = (b1+b2)/2.0
            bones[i][1, :] = (b1+b2)/2.0 #t[i, :]
        else:
            p = this.parent[i]
            # Points of the part at the interface with the parent
            b1 = np.mean(Pw[i][this.interfacePointsFromTo[i][p], :], axis=0)
            # Points of the parent at the interface with the part
            b2 = np.mean(Pw[p][this.interfacePointsFromTo[p][i], :], axis=0)
            # The joint is the mean of those points
            bones[i][0, :] = (b1+b2)/2.0
            c = this.children[i][0]
            if c > 0:
                b1 = np.mean(Pw[i][this.interfacePointsFromTo[i][p], :], axis=0)
                b2 = np.mean(Pw[i][this.interfacePointsFromTo[i][c], :], axis=0)
                bones[i][1, :] = (b1+b2)/2.0
            else:
                b1 = np.mean(Pw[i][this.interfacePointsFromTo[i][p], :], axis=0)
                b2 = np.mean(Pw[i], axis=0)
                bones[i][1, :] = b2
    return bones
