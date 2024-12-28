
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep import mpb

mp.verbosity(0)

n_core = 3.5 # Silicon
n_clad = 1.44 # SiO2
core = mp.Medium(index=n_core)
cladding = mp.Medium(index=n_clad)

resolution = 100 # pixels/μm

geometry_lattice = mp.Lattice(size=mp.Vector3(3,2))

w = 0.7  # width of waveguide
h = 0.25 # height of waveguide

geometry = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, mp.inf), material=cladding),
    # mp.Block(center=mp.Vector3(), size=mp.Vector3(w,h), material=core),
    mp.Prism(vertices = [mp.Vector3(-w/2,-h/2),mp.Vector3(w/2,-h/2),mp.Vector3(w/2-0.1,h/2),mp.Vector3(-w/2+0.1,h/2)],
        height = 0.1,
        axis = mp.Vector3(0,0,1),
        material=core)
           ]

wl0 = 1.55
freq = 1./wl0
num_modes = 3
ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_modes,
)

ms.init_params(mp.NO_PARITY, True)

eps = ms.get_epsilon()
x = np.arange(eps.shape[0]) / resolution
y = np.arange(eps.shape[1]) / resolution
plt.contourf(x,y,eps.transpose(), cmap = 'binary')
plt.colorbar(label = "Permittivity")
plt.savefig("Eps.png")
plt.close()


tol = 1e-5

n_mode_guess = 0.5 * (n_core + n_clad)
n_mode_min = n_clad
n_mode_max = n_core


k_guess = freq * n_mode_guess
k_min = freq * n_mode_min
k_max = freq * n_mode_max

neff = []
cmap = 'seismic'
for mode_num in range(1, num_modes + 1):
    k_mpb = ms.find_k(mp.NO_PARITY, freq, mode_num, mode_num, mp.Vector3(0,0,1), tol, k_guess, k_min, k_max)
    neff.append(k_mpb[0]/freq)
    E = ms.get_efield(which_band=mode_num)
    H = ms.get_hfield(which_band=mode_num)
    P = ms.get_poynting(which_band=mode_num)

    Ex = E[:,:,0,0]
    Ey = E[:,:,0,1]
    Ez = E[:,:,0,2]

    Pz = 0.5 * np.real(P[:,:,0,2])

    plt.contourf(x,y,np.abs(Ex.transpose()),202,cmap = cmap)
    plt.colorbar()
    plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(f"Ex_{mode_num}.png")
    plt.close()

    plt.contourf(x,y,Pz.transpose(),202,cmap = cmap)
    plt.colorbar()
    plt.xlabel("x"); plt.ylabel("y")
    plt.savefig(f"Poynting_{mode_num}.png")
    plt.close()

print(neff)
