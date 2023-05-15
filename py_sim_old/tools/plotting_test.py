import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import threading
import time


class Tester:
    """Example of how to use matplotlib with threading.

    Attributes:
        fig: Figure object for matplotlib
        ax: Axes object for matplotlib
        x: x values for matplotlib
        y: y values for matplotlib
        counter: counter for class
        lock: lock for matplotlib
    """

    def __init__(self):
        # Initialize figure and axes and save to class
        self.fig, self.ax = plt.subplots()
        # create initial values to plot
        self.plt_x = [i for i in range(5)]
        self.plt_width = [0.5 for i in range(5)]
        self.plt_counter = 0
        self._lock = threading.Lock()

    def plt_func(self, _):
        """Function for matplotlib animation.

        Args:
            _: Dummy variable for matplotlib animation

        Returns:
            Axes object for matplotlib
        """
        # lock thread
        with self._lock:
            x = np.array(self.plt_x)
            y = np.array(self.plt_width)
            self.ax.barh(y, 0.5, left=x, color="red")
            return self.ax

    def loop_logic(self):
        """Looping logic for adding data to plot."""
        time.sleep(3)
        print("init")
        while True:
            self.plt_add()
            time.sleep(1)

    def plt_add(self):
        """"Add data to class arrays."""
        with self._lock:
            if self.plt_counter > 4:
                self.plt_x.append(self.plt_counter)
                self.plt_width.append(0.5)
            self.plt_counter += 1
            print(self.plt_counter)

    def _plt(self):
        print("INIT PLOT")
        self.ani = anim.FuncAnimation(self.fig, self.plt_func, interval=1000)
        print("showing")
        plt.show()


if __name__ == "__main__":
    test = Tester()
    thread = threading.Thread(target=test.loop_logic)
    thread.start()
    test._plt()
