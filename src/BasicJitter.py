import numpy as np

class BasicJitter:
    def __init__(self):
        self.name = "Basic Jitter"

    def tophat(self, t1, t2, Fs = 1000, start_time = 0.0, end_time = 1.0):
        Ts = 1.0/Fs                      # sampling interval
        x = np.arange(0,Fs,1)
        t = np.arange(0,1,Ts)            # time vector
        
        x1 = x[t == t1]
        x2 = x[t == t2]
            
        nZeros = x1
        nPulse = x2 - x1 + 1

        y = np.zeros(nZeros)
        y = np.append(y, np.ones(nPulse))
        y = np.append(y, np.zeros(Fs-nPulse - nZeros))
        
        y = y[(t >= start_time) & (t <= end_time)]
        t = t[(t >= start_time) & (t <= end_time)]
        
        y = y/sum(y)
        
        return t, y

    def gaussian(self, t_mean, std_dev, Fs = 1000, start_time = 0.0, end_time = 1.0):

        Ts = 1.0/Fs                      # sampling interval
        x = np.arange(0,Fs,1)
        t = np.arange(0,1,Ts)            # time vector

        mean = x[t == t_mean]
        std_dev *= Fs
        
        y = (np.e**-(((x-mean)**2)/(2*std_dev**2)))/(std_dev*np.sqrt(2*np.pi))
        
        y = y[(t >= start_time) & (t <= end_time)]
        t = t[(t >= start_time) & (t <= end_time)]
        
        y = y/sum(y)
        
        return t, y

    def sine(self, mean, width, Fs = 1000, start_time = 0.0, end_time = 1.0):

        # Parameters for the sine wave distribution
        amplitude = width/2  # Amplitude of the sine wave
        frequency = 5.0  # Frequency of the sine wave (cycles per unit)
        phase = 0.0      # Phase shift of the sine wave
        num_samples = 10000  # Number of data points to generate

        # Generate data points from the sine wave distribution
        x = np.linspace(0, 2 * np.pi, num_samples)  # Create an array of evenly spaced values from 0 to 2*pi
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x + phase)  # Compute the sine wave values

        start = -0.5
        end = 0.5

        bins = np.linspace(start, end, int(((end-start)*Fs+1)))
        weightsa = np.ones_like(sine_wave)/float(len(sine_wave))
        freq, edge = np.histogram(np.array(sine_wave), bins, weights = weightsa)

        t = np.array(edge[:-1]) + mean
        y = np.array(freq)
        
        y = y[(t >= start_time) & (t <= end_time)]
        t = t[(t >= start_time) & (t <= end_time)]
        
        y = y/sum(y)
        
        return t, y