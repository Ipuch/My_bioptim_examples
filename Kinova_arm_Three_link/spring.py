import numpy as np
import matplotlib.pyplot as plt


def assignParam(springParam):
    S = yamaguchiSpring(springParam['s'],
                        springParam['k1'], springParam['k2'], springParam["q0"])
    return S


class yamaguchiSpring():
    def __init__(self, s: int, k1: float, k2, q0: float):
        """
        Parameters
        ----------
        s: 1 or -1
            sign
        k1: float
            stiffness
        k2: float
            stiffness
        q0: float
            coordinate at which there is no torque
        """
        if s is not None:
            self.s = s
        else:
            self.s = 1
        if k1 is not None:
            self.k1 = k1
        else:
            self.k1 = 0
        if k2 is not None:
            self.k2 = k2
        else:
            self.k2 = 0
        if q0 is not None:
            self.q0 = q0
        else:
            self.q0 = 0

    def torque(self, q):
        """
        Parameters
        ----------
        q: joint coordinate
        """
        s = self.s
        k1 = self.k1
        k2 = self.k2
        q0 = self.q0

        return s * k1 * np.exp(-k2 * (q - q0))

    def sign(self):
        return self.s

    def stiffness1(self):
        return self.s

    def stiffness2(self):
        return self.s

    def q0(self):
        return self.q0

# A = yamaguchiSpring(-1, -2, 10, 0)
# q = np.linspace(-0.1, 1.57, 100)
# plt.plot(q, A.torque(q))
# plt.plot(0, A.torque(0), 'o')
# plt.show()
