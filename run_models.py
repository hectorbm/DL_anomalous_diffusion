from models import two_state_switching_diffusion 
import matplotlib.pyplot as plt
import time

def main():
    x,y,state = two_state_switching_diffusion(n=500, k_state0=0.05, k_state1=0.01, D_state0=0.4, D_state1=0.005)
    #Values Tiempo Confinado aprox 0.005 normal 0.05 difusion confinado 0.001-0.01 normal 0.2-0.4 
    print(state)
    plt.plot(x,y)
    plt.show()
    for i in range(len(state)):
        if state[i] == 1:
            plt.scatter(x[i],y[i],color='b',label=("Blue:State1","Red:State0"))
        else:
            plt.scatter(x[i],y[i],color='r',label=("Blue:State1","Red:State0"))
    plt.show()



if __name__ == "__main__":
    main()


