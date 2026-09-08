"""Fixed pool of decorative proximity flashes, not stellar explosions.

Detection is sampled; existing flashes retain their positions between passes.
A detection batch is spawned once, never replayed on intermediate frames.
Pool update and spawn allocate no ndarrays; detection and rendering may.
"""
import numpy as np
import config as C
import collisions


class FlashPool:
    def __init__(self):
        self.pos = np.zeros((C.MAX_FLASHES, 3))
        self.age = np.full(C.MAX_FLASHES, C.FLASH_LIFETIME)
        self.peak = np.zeros(C.MAX_FLASHES)
        self.brightness = np.zeros(C.MAX_FLASHES)
        self.colour = np.tile([1.0, 0.0, 0.65], (C.MAX_FLASHES, 1))
        self._rise = np.zeros(C.MAX_FLASHES)
        self._decay = np.zeros(C.MAX_FLASHES)
        self._next = 0

    def clear(self):
        self.age.fill(C.FLASH_LIFETIME)
        self.peak.fill(0.0)
        self.brightness.fill(0.0)
        self._next = 0

    def spawn(self, pos, pairs):
        for pair in range(len(pairs)):
            # All slots age together: FIFO is oldest-first, even for age ties.
            slot = self._next
            self._next = (slot + 1) % C.MAX_FLASHES
            i, j = int(pairs[pair, 0]), int(pairs[pair, 1])
            for axis in range(3):
                self.pos[slot, axis] = pos[i, axis]*0.5 + pos[j, axis]*0.5
            self.age[slot] = 0.0
            self.peak[slot] = C.FLASH_PEAK

    def update(self, wall_dt):
        # Every array-producing ufunc has an explicit preallocated output.
        np.add(self.age, max(0.0, wall_dt), out=self.age)
        np.minimum(self.age, C.FLASH_LIFETIME, out=self.age)
        np.divide(self.age, C.FLASH_RISE, out=self._rise)
        np.minimum(self._rise, 1.0, out=self._rise)
        np.subtract(C.FLASH_LIFETIME, self.age, out=self._decay)
        np.divide(self._decay, C.FLASH_LIFETIME-C.FLASH_RISE, out=self._decay)
        np.clip(self._decay, 0.0, 1.0, out=self._decay)
        np.multiply(self._rise, self._decay, out=self.brightness)
        np.multiply(self.brightness, self.peak, out=self.brightness)


class CollisionEffects:
    def __init__(self):
        self.pool = FlashPool()
        self.frame = 0

    def clear(self):
        self.pool.clear()
        self.frame = 0

    def advance(self, pos, wall_dt, *, sample=True):
        self.pool.update(wall_dt)
        if sample:
            if self.frame % C.COLLISION_INTERVAL == 0:
                self.pool.spawn(pos, collisions.detect(pos))
            self.frame += 1
