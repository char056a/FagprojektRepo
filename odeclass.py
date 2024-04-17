class ODE:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)

        if not getattr(self, "state_keys",0):
            self.state_keys = list(data.keys())
        if not getattr(self, "timestep",0):
            self.timestep = 1
        for key in self.state_keys:
            setattr(self, key+"0", data[key])
    def __str__(self):
        return str(self.__dict__)
        
    def get_state(self):
        """Update state vector to values given by input"""
        x = np.array([getattr(self,key) for key in self.state_keys])
        return x

    def get_attr(self, states, attr):
        """Returns column of state matrix with idx matching given attribute"""
        idx = self.state_keys.index(attr) # Finds the index of desired attribute
        return states[:,idx]

    def update_state(self, x_new):
        """Update state vector to values given by input"""
        for key, val in zip(self.state_keys, x_new):
            setattr(self, key, val)
        return

    def reset(self):
        """Resets state to x0"""
        x0 = np.array([getattr(self,key+"0") for key in self.state_keys])
        self.update_state(x0)
        return

    def time_arr(self, length):
        return np.arange(0, length*self.timestep, self.timestep)

    def euler_step(self, dx):
        """
        Updates state using state vector derivative and one step of eulers method.
        
        Parameters
        ----------
        dx : numpy array
            Derivative of state vector.
        """
        x_new = self.get_state() + dx * self.timestep
        self.update_state(x_new)
        return
