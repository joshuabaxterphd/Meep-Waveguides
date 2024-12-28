
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Simulation domain size
Size_x = 3
Size_y = 10
Size_z = 2
cell_size = mp.Vector3(Size_x,Size_y,Size_z)

resolution = 25 # cells per micron
pml_size = 10 / resolution  # 10 FDTD cells for PML
pml_layers = [mp.PML(thickness=pml_size)]

# # Waveguide materials
n_core = 3.5 # Silicon
n_clad = 1.44 # SiO2
core = mp.Medium(index=n_core)
cladding = mp.Medium(index=n_clad)

w = 0.5 # waveguide width
h = 0.22 # waveguide height

# # Waveguide
geometry = [
    mp.Block(center=mp.Vector3(), size=mp.Vector3(w, mp.inf, h), material=core),
           ]

# Setup source wavelenghts/frequencies
wl0 = 1.55
min_wl = 1.4
max_wl = 1.7

freq = 1./wl0
min_freq = 1 / max_wl
max_freq = 1 / min_wl
f_width = max_freq - min_freq
pulse = mp.GaussianSource(freq,fwidth=f_width * 2) # gaussian pulse

source_pos_y = - Size_y // 2 + pml_size + 0.1

# Mode source
sources = [mp.EigenModeSource(pulse,
                              center=mp.Vector3(y = source_pos_y),
                              size=mp.Vector3(x = Size_x, z = Size_z),
                              direction=mp.Y,
                              eig_band=1,
                              )]

# Simulation object
sim = mp.Simulation(cell_size=cell_size,
                    resolution=resolution,
                    boundary_layers=pml_layers,
                    sources=sources,
                    geometry=geometry)


# DFT Monitor
dft_freqs = [1/min_wl,1/wl0,1/max_wl]
dft_fields = sim.add_dft_fields([mp.Ex, mp.Ez],
                                dft_freqs,
                                center = mp.Vector3(),
                                size = mp.Vector3(Size_x,Size_y))


# Flux monitors for mode decomposition
monitor_wavelengths = np.linspace(min_wl,max_wl,31)
freqs = 1./monitor_wavelengths
inc_loc = source_pos_y + 0.2
trans_loc = Size_y/2 - pml_size - 0.2
inc = sim.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(y = inc_loc), size=mp.Vector3(x = Size_x, z = Size_z)))
trans = sim.add_flux(freqs, mp.FluxRegion(center=mp.Vector3(y = trans_loc), size=mp.Vector3(x = Size_x, z = Size_z)))


sim.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,Size_y)))
plt.savefig("sim.png")
plt.close()

sim.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(Size_x,0,Size_z)))
plt.savefig("simXZ.png")
plt.close()

sim.run(until_after_sources = mp.stop_when_fields_decayed(5, mp.Ez, mp.Vector3(0,Size_y/2 - pml_size - 0.1), 1e-4))


# Do Mode decomposition
trans_ = sim.get_eigenmode_coefficients(trans,
                                     [1,2],
                                     direction=mp.Y,
                                     )
inc_ = sim.get_eigenmode_coefficients(inc,
                                     [1,2],
                                     direction=mp.Y,
                                     )


aInc = inc_.alpha[0,:,0]
aT = trans_.alpha[0,:,0]
aR = inc_.alpha[0,:,1]

aInc2 = inc_.alpha[1,:,0]
aT2 = trans_.alpha[1,:,0]
aR2 = inc_.alpha[1,:,1]
print(inc_.alpha.shape)

# Calculate and plot transmittance and reflectance
T = np.abs(aT) ** 2 / np.abs(aInc) ** 2
R = np.abs(aR) ** 2 / np.abs(aInc) ** 2

T2 = np.abs(aT2) ** 2 / np.abs(aInc) ** 2
R2 = np.abs(aR2) ** 2 / np.abs(aInc) ** 2


plt.plot(monitor_wavelengths,T,"red")
plt.plot(monitor_wavelengths,T2,'x',color = "red")
plt.plot(monitor_wavelengths,R,"blue")
plt.plot(monitor_wavelengths,R2,'x',color = "blue")
plt.xlabel("wavelengths (um)")
plt.legend(["T","T2","R","R2"])
plt.savefig("TR.png")
plt.close()


# loop through frequencies, and plot DFT fields
for f in range(len(dft_freqs)):

    Ex_f = sim.get_dft_array(dft_fields,mp.Ex,f)
    Ez_f = sim.get_dft_array(dft_fields,mp.Ez,f)

    plt.imshow(np.real(Ex_f),extent = [-Size_y/2, Size_y/2, -Size_x/2, Size_x/2])
    plt.colorbar()
    plt.savefig(f"Ex_f_{1./dft_freqs[f]}.png")
    plt.close()

    plt.imshow(np.real(Ez_f),extent = [-Size_y/2, Size_y/2, -Size_x/2, Size_x/2])
    plt.colorbar()
    plt.savefig(f"Ez_f_{1./dft_freqs[f]}.png")
    plt.close()
