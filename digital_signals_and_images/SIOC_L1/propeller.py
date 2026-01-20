import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio as iio

M = 64
n = 5
l = 16
K = 256/l

gif_path = 'propeller.gif'

def generate_propeller():

    for m in range(-M//2, M//2):
        # obliczenia współrzędnych biegunowych
        theta = np.linspace(0, 2*np.pi, 500)
        r = np.sin(3*theta + m*np.pi/10)
        
        # przekształcenie na współrzędne kartezjańskie
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        plt.clf()
        plt.plot(x, y, color='blue')
        plt.grid(True)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.title('Propeller Animation')
        plt.savefig(f'frames/frame_{m + M//2:03d}.jpg')

    frames = [iio.imread(f'frames/frame_{i:03d}.jpg') for i in range(M)]

    iio.mimsave(gif_path, frames)
        

generate_propeller()
