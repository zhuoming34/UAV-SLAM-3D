import numpy as np
import math
from filterpy.kalman import predict, update
from collections import namedtuple
import kf_book.kf_internal as kf_internal
from kf_book.kf_internal import DogSimulation

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '(mean={:.3f}, var={:.3f})'.format(s[0], s[1])

np.random.seed(13)

def height(h, std):
    return h + (randn() * std)

height_std = .13
process_var = .05**2
actual_height = 16.3
x = gaussian(25., 1000.) # initial state
process_model = gaussian(0., process_var)
N = 50
zs = [height(actual_height, height_std) for i in range(N)]
ps = []
estimates = []
for z in zs:
prior = predict(x, process_model)
x = update(prior, gaussian(z, voltage_std**2))
# save for latter plotting
estimates.append(x.mean)
ps.append(x.var)

book_plots.plot_measurements(zs)
book_plots.plot_filter(estimates, var=np.array(ps))
book_plots.show_legend()
plt.ylim(16, 17)
book_plots.set_labels(x='step', y='volts')
plt.show()
plt.plot(ps)
plt.title('Variance')
print('Variance converges to {:.3f}'.format(ps[-1]))

"""
process_var = 1.  # variance in the dog's movement
sensor_var = 1.  # variance in the sensor

x = gaussian(50., 1. ** 2)  # dog's position, N(0, 20**2)
velocity = 0
dt = 1.  # time step in seconds
process_model = gaussian(velocity * dt, process_var)  # displacement to add to x

# simulate dog and get measurements
dog = DogSimulation(
    x0=x.mean,
    velocity=process_model.mean,
    measurement_var=sensor_var,
    process_var=process_model.var)
# create list of measurements
zs = [dog.move_and_sense() for _ in range(500)]

print('PREDICT\t\t\tUPDATE')
print('     x      var\t\t  z\t    x      var')

# perform Kalman filter on measurement z
for z in zs:
    prior = predict(x, process_model)
    likelihood = gaussian(z, sensor_var)
    x = update(prior, likelihood)

    kf_internal.print_gh(prior, x, z)

print()
print('final estimate:        {:10.3f}'.format(x.mean))
print('actual final position: {:10.3f}'.format(dog.x))
"""

# -----------------------


"""
def std_kf(x, P, u, Q, z, R):
    # x, P: state and variance of the system
    # u, Q: movement due to the process, and noise in the process
    # z, R: measurement and measurement variance
    for i in range(50):
        x, P = predict(x=x, P=P, u=u, Q=Q)
        x, P = update(x=x, P=P, z=z, R=R)
    return x

if __name__ == '__main__':
    x = 10.
    P = 0.1
    u = 0.0
    Q = 0.1
    z = 0.0
    R = 0.1

    result = std_kf(x, P, u, Q, z, R)

    print(result)
"""