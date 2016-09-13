from .discrete import baseline
from .algorithms import baseline as dt_baseline
import matplotlib.pyplot as plt
import numpy as n
from os.path import dirname, join

from uediff.diffsim import powder_diffraction
from uediff.instrumentation import gaussian
from uediff.structure import InorganicCrystal

def generate_data():
    VO2 = InorganicCrystal.from_preset('M1')
    s, I = powder_diffraction(crystal = VO2, plot = False, scattering_length = n.linspace(0.11, 0.8, 1000))
    I -= dt_baseline(array = I, max_iter = 1000, background_regions = [slice(0, 29)])
    n.save(join(dirname(__file__), 'data', 'diffraction.npy'), n.vstack((s.flatten(), 20*I.flatten())) )

def load_data(filename = 'diffraction.npy'):
    return n.load(join(dirname(__file__), 'data', filename))

def test_baseline():
    scatt_angle = n.linspace(0.11, 0.8, 1000)
    noise = n.random.random(size = scatt_angle.shape)

    amp, subamp, dec, subs1, subs2 = 75, 55, -7, 0.8, 1
    background = amp*n.exp(dec*scatt_angle) + subamp*n.exp(-2*scatt_angle) + subs1*gaussian(scatt_angle, (scatt_angle.max() + scatt_angle.min())/2, (scatt_angle.max() - scatt_angle.min())/8) + subs2*gaussian(scatt_angle, (scatt_angle.max() + scatt_angle.min())/2.5, (scatt_angle.max() - scatt_angle.min())/8)

    _, I = load_data()

    disc_baseline = baseline(array = I + background + noise, max_iter = 100, wavelet = 'sym6', background_regions = [slice(0, 29)])
    dual_baseline = dt_baseline(array = I + background + noise, max_iter = 100, background_regions = [slice(0, 29)])
    
    plt.plot(scatt_angle, I + background + noise, '.k')
    plt.plot(scatt_angle, disc_baseline, '.b')
    plt.plot(scatt_angle, dual_baseline, '.r')
    plt.show()

if __name__ == '__main__':
    test_baseline()