from src.BasicJitter import BasicJitter

import matplotlib.pyplot as plt

def main ():
    basicjitter = BasicJitter()
    print(basicjitter.name)

    mean = 0.5
    Fs = 200
    width_tophat = 0.2

    t_tophat, y_tophat = basicjitter.tophat(mean - width_tophat/2, mean + width_tophat/2, Fs)
 
    plt.plot(t_tophat,y_tophat,'k-')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title("Mean: {}, Width: {}".format(mean, width_tophat))
    image_dir = 'results/figures/'
    plt.savefig(image_dir + "tophat.png", format = "png")

    # plt.show()
    
if __name__ == '__main__':
    main()