import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
# import tifffile as tiff
from PIL import Image

output_path = '/home/agoering/agoering/particle_singles'

nP = 5000
nX = 500
nY = 500
x = range(nX)
y = range(nY)
X, Y = np.meshgrid(x,y)

d = 6
I = 255
r = d*2

rng = np.random.default_rng(seed=42)

ti = time.perf_counter()
for j in tqdm(range(10000)):
    #Generate particle positions
    ps = rng.random(size=(nP,2))
    ps[:,0] = ps[:,0]*nX
    ps[:,1] = ps[:,1]*nY

    im = np.zeros((nY,nX))

    for i in range(nP):
        p = ps[i,:]
        x = p[0]
        xr = round(x)
        y = p[1]
        yr = round(y)
        ymin = max(yr - r,0)
        ymax = min(yr + r, nY-1)
        xmin = max(xr - r,0)
        xmax = min(xr + r, nX-1)
        im[ymin:ymax, xmin:xmax] = im[ymin:ymax, xmin:xmax] + I*np.exp((-(X[ymin:ymax, xmin:xmax]-x)**2-(Y[ymin:ymax, xmin:xmax]-y)**2) / (1/8*d**2))


    dir = Path(output_path)
    dir.mkdir(exist_ok=True)

    # plt.figure()
    # f1 = plt.imshow(im, cmap='gray')
    # plt.colorbar(f1)
    # plt.savefig(dir / f"{j}.tif")
    # plt.close()

    # tiff.imwrite(dir / f"{j}.tif",im)
    Im = Image.fromarray(im)
    Im.save(dir / f"{j}.tif")

tf = time.perf_counter()
print(f'time elapsed: {tf-ti} s')