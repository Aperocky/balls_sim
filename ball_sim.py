import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
A simulation of how lighter element will come to dominate the upper atmosphere

Written in boredom while in flight from Toronto to Beijing

Author, Rocky Li
zl131@duke.edu

05/16/2018 Airborne

Additional credit to
Jake Vanderplas
vanderplas@astro.washington.edu

He came up with some of the underlying logic which I consulted.
"""

# Simulation class
class Box:

    # The box is enlongated altitude wide for idea illustration
    def __init__(self, dt, borders= 800, gforce = False):
        self.borders = [0, borders, 0, borders*2]
        self.dt = dt
        self.gforce = gforce

    # APPLY G-FORCE
    def apply_gforce(self):
        self.particles[:, 3] += self.dt* -100

    # This set the particles independent of the class method - have any beginning configuration you want.
    """ IMPORTANT : Particles format (posX, posY, velX, velY, size(radius), mass) """
    def set_particles(self, particles):
        self.particles = np.asarray(particles)
        self.numparticle = len(particles)

    # Generate particles for the heavier elements. The heavier elements are 5 by mass and radius
    def gen_particles(self, numparticle):
        self.numparticle = numparticle
        locations = np.random.uniform(10, self.borders[1]-10, (numparticle,2))
        velocities = np.random.uniform(-100, 100, (numparticle,2))
        sizemass = np.ones((numparticle,2))*5
        self.particles = np.concatenate((locations, velocities, sizemass), axis=1)

    # Generate lighter particles, the elements are 5 by radius and 1 by mass
    def lighterelement(self, numparticle):
        self.numparticle += numparticle
        locations = np.random.uniform(10, self.borders[1]-10, (numparticle,2))
        velocities = np.random.uniform(-100, 100, (numparticle,2))
        sizemass = np.ones((numparticle,2))
        sizemass[:, 0] *= 5
        appendage = np.concatenate((locations, velocities, sizemass), axis=1)
        self.particles = np.concatenate((self.particles, appendage), axis=0)

    # Run the simulation based on time stepping dt.
    def run(self):
        self.particles[:, :2] += self.dt * self.particles[:, 2:4]
        if self.gforce:
            self.apply_gforce()
        collist = self.get_collision()
        for indexi, indexj in collist:
            self.collision(indexi, indexj)
        self.bordercheck()
        # print(self.particles)

    # Get a list of collision that's going on
    def get_collision(self):

        # Get the collision range of all particles with E/O
        def getsizematrix():
            leng = self.numparticle
            colrange = np.zeros((leng,leng))
            radius = self.particles[:,4]
            colrange[:] += radius
            for i in range(leng):
                colrange[:,i] += radius
            return colrange

        # Get the distance matrix of all particles within the matrix
        def getdistmatrix():
            dists = squareform(pdist(self.particles[:,:2]))
            return dists

        def getcollisions(colrange, dists):
            indexi, indexj = np.where(dists < colrange)
            uniques = indexi < indexj
            indexi = indexi[uniques]
            indexj = indexj[uniques]
            return zip(indexi, indexj)

        colrange = getsizematrix()
        dists = getdistmatrix()
        collist = getcollisions(colrange, dists)
        return collist

    # This solves a SINGLE collision with no energy loss
    def collision(self, indexi, indexj):
        massi = self.particles[indexi][-1]
        massj = self.particles[indexj][-1]

        posi = self.particles[indexi][:2]
        posj = self.particles[indexj][:2]

        veli = self.particles[indexi][2:4]
        velj = self.particles[indexj][2:4]

        col_size = self.particles[indexi][-2] + self.particles[indexj][-2]

        # relative location & velocity vectors
        r_rel = posi - posj
        v_rel = veli - velj
        r_dist = np.sqrt(np.sum(r_rel**2))

        # Can't be inside each other.
        posi += r_rel*(1-r_dist/col_size)/2
        posj -= r_rel*(1-r_dist/col_size)/2

        # momentum vector of the center of mass
        v_cm = (massi * veli + massj * velj) / (massi + massj)

        # collisions of spheres reflect v_rel over r_rel
        rr_rel = np.dot(r_rel, r_rel)
        vr_rel = np.dot(v_rel, r_rel)
        v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

        # Change the dots
        self.particles[indexi, 2:4] = v_cm + v_rel * massj / (massi + massj)
        self.particles[indexj, 2:4] = v_cm - v_rel * massi / (massi + massj)

    # Bounce back when particles hits wall.
    def bordercheck(self):
        crossed_x1 = (self.particles[:, 0] < self.borders[0] + self.particles[:, -2])
        crossed_x2 = (self.particles[:, 0] > self.borders[1] - self.particles[:, -2])
        crossed_y1 = (self.particles[:, 1] < self.borders[2] + self.particles[:, -2])
        crossed_y2 = (self.particles[:, 1] > self.borders[3] - self.particles[:, -2])

        self.particles[crossed_x1, 0] = self.borders[0] + self.particles[crossed_x1, -2]
        self.particles[crossed_y1, 1] = self.borders[2] + self.particles[crossed_y1, -2]
        self.particles[crossed_x2, 0] = self.borders[1] - self.particles[crossed_x2, -2]
        self.particles[crossed_y2, 1] = self.borders[3] - self.particles[crossed_y2, -2]

        self.particles[crossed_x1, 2] *= -1
        self.particles[crossed_y1, 3] *= -1
        self.particles[crossed_x2, 2] *= -1
        self.particles[crossed_y2, 3] *= -1

# Defines a visual for viewing the above Box class
class Vision:

    # Vision for the simulation
    def __init__(self, box):
        self.box = box
        self.figure = plt.figure()
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.figure.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(box.borders[:2]), ylim=(box.borders[2:]))
        ms = int(self.figure.dpi * 5 * self.figure.get_figwidth()
                 / np.diff(self.ax.get_xbound())[0])
        self.particles, = self.ax.plot([], [], 'bo', ms=ms)
        self.particleslight, = self.ax.plot([], [], 'ro', ms=ms)

    def init(self):
        self.particles.set_data(box.particles[:1000, 0], box.particles[:1000, 1])
        return self.particles

    def update(self, i):
        self.box.run()
        self.particles.set_data(box.particles[:1000, 0], box.particles[:1000, 1])
        self.particleslight.set_data(box.particles[1000:, 0], box.particles[1000:, 1])
        ms = int(self.figure.dpi * 2.5 * self.figure.get_figwidth()
                 / np.diff(self.ax.get_xbound())[0])
        self.particles.set_markersize(ms)
        self.particleslight.set_markersize(ms)
        return self.particles

    def animate(self):
        ani = animation.FuncAnimation(self.figure, self.update, frames=600, interval=20, init_func=self.init)
        plt.show()

if __name__ == "__main__":
    box = Box(0.05, gforce=True)
    box.gen_particles(1000)
    box.lighterelement(200)
    vision = Vision(box)
    vision.animate()
