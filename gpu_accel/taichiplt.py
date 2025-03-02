import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

pixels = ti.field(ti.f32, (512, 512))

def render_pixels():
    arr = np.random.rand(512, 512).astype(np.float32)
    pixels.from_numpy(arr)   # load numpy data into taichi fields

render_pixels()
arr = pixels.to_numpy()  # store taichi data into numpy arrays
plt.imshow(arr)
plt.show()
import matplotlib.cm as cm
cmap = cm.get_cmap('magma')
gui = ti.GUI('Color map', (512, 512))

while gui.running:
    render_pixels()
    arr = pixels.to_numpy()
    gui.set_image(cmap(arr))
    gui.show()