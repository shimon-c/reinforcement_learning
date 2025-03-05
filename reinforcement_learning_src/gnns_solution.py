import numpy as np

"""
4 satalites in location as below 
x≈4632.68 km,y≈602.87 km,z≈4197.01 km

"""



satalites= [
    #each row [x,y,z], d
    [(15600,7540,20140),20432],
    [(18760,2750,18610), 21045],
    [(17610,14630,13480), 21189],
    [(19170,610,18390),19729]
]

def solve_lin_step(sat_locs, xo=None,yo=None,zo=None,tau=0,C=300000):
    N = sat_locs.shape[0]
    A = np.zeros((N,4))

    A[:,0] = (xo-sat_locs[:,0])/sat_locs[0,-1]
    A[:, 1] = (yo-sat_locs[:, 1]) / sat_locs[1,-1]
    A[:, 2] = (zo-sat_locs[:, 2]) / sat_locs[2,-1]
    A[:, 3] = C
    A_inv = np.linalg.pinv(A[:,:3])
    cur_xo = np.array([xo,yo,zo])
    D = sat_locs[:,:3] - cur_xo
    D2 = D * D
    D2_S = np.sum(D2, axis=1)
    D2_sqrt = np.sqrt(D2_S)
    del_vals = sat_locs[:,3] - D2_sqrt
    del_vec = A_inv @ del_vals

    return del_vec

def solve_lin():
    N = len(satalites)
    A = np.zeros((N, 4))
    sat_locs = np.zeros((N,4))
    for k in range(N):
        cur_loc = satalites[k][0]
        sat_locs[k,:3] = np.array(cur_loc)
        dst = satalites[k][-1]
        sat_locs[k,-1] = dst

    xo = np.mean(sat_locs[:,0])
    yo = np.mean(sat_locs[:,1])
    zo = np.mean(sat_locs[:,2])
    for k in range(10000):
        del_vals = solve_lin_step(sat_locs=sat_locs, xo=xo, yo=yo,zo=zo)
        xo += del_vals[0]
        yo += del_vals[1]
        zo += del_vals[2]
        print(f'del_vec:{del_vals}')

solve_lin()


