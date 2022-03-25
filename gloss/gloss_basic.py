"""

    Author: Silvia Zuffi

    Functions for basic computations and visualization with the body model.

"""

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import cv2

def get_part_mesh(this, part, Zp=None, Zs=None, has_pose_space=True):
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

    if has_pose_space:
        # Add mean pose deformations
        M = this.posePCA[part]['M']+Tp

        # Add pose deformations
        B = this.posePCA[part]['B']

    if Zp is None or not has_pose_space:
        A = Tp
    else:
        nB_p = len(Zp)
        A = M + np.array(B[:, 0:nB_p]*np.matrix(Zp).T)[:, 0]

    P = np.zeros((int(len(A)/3.), 3))
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


