"""
"""

import numpy as np
import pdb
from psbody.mesh.meshviewer import MeshViewer
from psbody.mesh.mesh import Mesh
import gloss.stitched_skel as gloss_animal
from gloss.gloss_get_sample_from_model import get_sample
from gloss.gloss_core.gloss_skeleton import GlossSkeleton

def show_shape_samples(N=1):
    mv = MeshViewer()
    #mv.set_background_color(np.ones(3))
    #seed = 2
    #np.random.seed(seed)

    #bm = gloss_animal.StitchedSkeleton()
    #model = './models/skeleton_gloss_no_fingers.pkl'
    #bm.load_model(model)


    model_name = 'skeleton_gloss_no_fingers'
    model_dir='./models/'
    bm = GlossSkeleton(model_dir=model_dir, animal_name=model_name, data_dir=model_dir)
    bm.load_model(model_name)

    # Uncomment to see the shape space
    #bm.show_model(bm.shapePCA, add_cage=False, forPart=None, nPC=7,
    #              nSigma=3, in_color=None, stop=True)


    i = 0

    D = bm.parts

    for i in range(N):
        P, r_abs, t, Zp, zs = get_sample(bm,
                nB_pose=None,
                nB_shape=7,
                fixedShape=False,
                add_global_rotation=False,
                multi_shape_PCA=False, 
                init_rotation_axis=None, has_pose_space=False)

        for part in bm.partSet:
            P[part][:,0] = P[part][:,0] + 1.5*i

        ms = [Mesh(v=P[part], f=bm.partFaces[part]).set_vertex_colors(bm.colors[part]) for part in bm.partSet]

        if i == 0:
            M = ms 

        else:
            for part in bm.partSet:
                M.append(Mesh(v=P[part], f=bm.partFaces[part]).set_vertex_colors(bm.colors[part])) 
    mv.set_static_meshes(M)
    pdb.set_trace()




if __name__ == '__main__':
    for i in range(10):
        show_shape_samples(N=1)
