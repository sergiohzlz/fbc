#!/usr/bin/python
#coding:utf8

import os
import sys, getopt

from scipy.stats      import linregress
from scipy.optimize   import curve_fit
from numpy            import exp, log, log10, array,  power, arange, loadtxt, sqrt, diag
from numpy.random     import normal, random, choice
from sklearn.metrics  import r2_score

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Germibeta(object):

    def carga_archivo(self, archivo, sep=","):
        """
        Carga el archivo en el parámetro
        """
        self.f = loadtxt(archivo,delimiter=sep)
        self.N = len(self.f)
    
    def germibeta(self, 
                  r : int,
                  alfa : float, 
                  beta : float, 
                  A, 
                  N, base=10) -> float:
        """ 
        Fase final para hacer el cálculo de  la 
        distribución usando los valores en los parámetros

        Parameters:
        r       : arreglo de enteros
        alfa    : exponente del denominador
        beta    : exponente del numerador
        A       : Constante de normalización
        N       : entero con la cantidad de rangos
        base=10
        Returns:
        Lista de valores con los valores asignados en el parámetro
        """
        fac = base**A
        num = power((N+1-r),beta)
        den = power(r,alfa)
        return fac*num/den

    def genera_x0(self, verbose=True):
        """
        Toma la distribución F y genera a través de una
        regresión lineal el punto x0 que será usado para
        otros métodos.
        """
        N = len(F)
        R = range(1,N+1)
        # r = arange(1,(N+1), 0.01)
        lgR, lgF = log10(R), log10(F)
        V = linregress(lgR, lgF)
        if(verbose):
            print(str(V))
        m = abs(V.slope)
        b = abs(V.intercept)
        return array([b, abs(m), abs(m)])

    def ajuste(self, F, verbose=False):
        """
        Ajusta no-lineal de los datos en F
        Se usa Levenberg-Marquadt para el ajuste
        """
        N = len(F)
        R = range(1,N+1)
        r = arange(1,(N+1), 0.01)
        lgR, lgF = log10(R), log10(F)
        V = linregress(lgR, lgF)
        if(verbose):
            print(str(V))
        m = abs(V.slope)
        b = abs(V.intercept)

        x0 = array( [b, abs(m), abs(m)] )
        modelo = lambda r,x,y,z : germibeta(r, x, y, z, N)
        popt, pcov = curve_fit( modelo, R, F, x0, sigma=F, method='lm')
        r2 = r2_score( F, modelo(array(R), popt[0], popt[1], popt[2]) )
        if(verbose):
            print(str(popt))
            print(str(sqrt(diag(pcov))))
            print(str(r2))
        return popt , pcov, r2


def germibeta(r, alfa, beta, A, N, base=10):
    fac = base**A
    num = power((N+1-r),beta)
    den = power(r,alfa)
    return fac*num/den

def genera_individuos(largo,n):
    """
    GA.
    Genera N individuos con un cromosoma de largo=largo
    """
    cromosoma = lambda largo: [choice([0,1]) for i in range(largo)]
    P = []
    for i in range(n):
        P.append(cromosoma(largo))
    return P

def genera_x0(F):
    """
    Toma la distribución F y genera a través de una
    regresión lineal el punto x0 que será usado para
    otros métodos.
    """
    N = len(F)
    R = range(1,N+1)
    r = arange(1,(N+1), 0.01)
    lgR, lgF = log10(R), log10(F)
    V = linregress(lgR, lgF)
    # if(verbose):
    #     print(str(V))
    m = abs(V.slope)
    b = abs(V.intercept)
    return array([b, abs(m), abs(m)])

def ajuste(F, verbose=False):
    """
    Ajusta no-lineal de los datos en F
    Se usa Levenberg-Marquadt para el ajuste
    """
    N = len(F)
    R = range(1,N+1)
    r = arange(1,(N+1), 0.01)
    lgR, lgF = log10(R), log10(F)
    V = linregress(lgR, lgF)
    if(verbose):
        print(str(V))
    m = abs(V.slope)
    b = abs(V.intercept)

    x0 = array( [b, abs(m), abs(m)] )
    modelo = lambda r,x,y,z : germibeta(r, x, y, z, N)
    popt, pcov = curve_fit( modelo, R, F, x0, sigma=F, method='lm')
    r2 = r2_score( F, modelo(array(R), popt[0], popt[1], popt[2]) )
    if(verbose):
        print(str(popt))
        print(str(sqrt(diag(pcov))))
        print(str(r2))
    return popt , pcov, r2

def graf_datos(y, arr, titulo, nomf):
    """
    Grafica los datos en y
    y el ajuste en arr
    """
    a,b,A,N,r2 = arr
    N = int(N)
    R = arange(1,N+1,0.05)
    params = [R,a,b,A,N]
    Y = germibeta(*params)
    fig = plt.figure()
    plt.semilogy(range(1,N+1),y,'.', R,Y)
    plt.xlabel(r'\textbf{Rango}')
    plt.ylabel(r'\textbf{Frecs (log)}')
    plt.title(titulo + '\n' + r"$(\alpha,\beta)$=({0:.2f},{1:.2f}), $r^2$={2:.4f}: ".format(a,b,r2))
    plt.savefig(nomf+'.png')
    del(fig)


def uso():
    print("Ejemplo\npython germibeta -i archivo -c columna -m lineas -v|--verbose")


if __name__ =='__main__':
    try:
        args = sys.argv[1:]
        opts, args = getopt.getopt( args, "i:vc:m:s:", ["input=","muestreo=","verbose","columna=","sep="] )
    except getopt.GetoptError:
        print("Erorr en los parámetros")
    verbose = False
    columna  = -1
    umbral   = 1
    sep      = ","
    for opt,arg in opts:
        if opt in ('-i','--input'):
            archivo = arg
            if(os.path.exists(archivo)):
                print(f"Usando {archivo}")
            else:
                print("No se encuentra el archivo")
        elif opt in ('-v','--verbose'):
            verbose = True
        elif opt in ('-c','--columna'):
            if(int(arg)>0): columna = int(arg)-1
        elif opt in ('-m','--muestreo'):
            umbral = float(arg)
        elif opt in ('-s','--sep'):
            sep = arg

    if(verbose):
        print("Usando")
        print("Archivo {0}".format(archivo))
        if(columna != -1):
            print("Usando columna {0} de archivo {1}".format(str(columna),archivo))
        else:
            print("Usando columna 1 de archivo {0}".format(archivo))
        if(umbral != 1):
            print("Porcentaje de muestreo {0}".format(umbral))
    f = loadtxt(archivo,delimiter=sep)
    N = len(f)

    if verbose: print("Líneas leídas {0}".format(N))
    if(columna==0):
        popt, pcov, r2 = ajuste(f)
    elif(columna>0):
        popt, pcov, r2 = ajuste(f[:,columna])
    alfa, beta = popt[:2]
    A = popt[2]
    error = sqrt(diag(pcov))
    if(verbose):
        print("Resultado")
        print("Archivo {0} con {1} rangos".format(archivo,N))
        print("A {0}".format(A))
        print("alfa, beta : ({0},{1})".format(alfa,beta))
        print("R2 {0}".format(r2))
        print("Error {0}".format(error))
    else:
        V = ','.join(map(str,[ archivo, A, alfa, beta, N, r2]))
        print(V)


