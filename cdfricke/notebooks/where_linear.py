# Programmers: Connor Fricke (cd.fricke23@gmail.com) Jayde Spiegel (jdspiegel1221@gmail.com)
# File: where_linear.py
# Latest Rev: 18-Mar-2025
# Desc: module for linear domain finder, designed to find a region within data which is best fit by linear regression

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def line(x, m, b):
    return m * x + b

class Domain:
    """
    Object for storing information about different domains within the data.
    The goal is to find the largest linear domain. If there are two domains
    with an equal size, the preferential domain is the one with the larger slope,
    reflected in the overload of the __gt__ and __lt__ operators.
    Primarily, this is simply a helper class for the LinearDomainFinder class.

    @@ data members:
    id: int - identification number for a particular domain
    shift: int - the index of the first value in the data corresponding to the domain
    size: int - the number of consecutive data points included in the domain
    slope: float - the slope of the fit generated using all data points in the domain
    """
    def __init__(self):
        self.id: int = 0
        self.shift: int = 0
        self.size: int = 0
        self.slope: float = 0.0
    def __init__(self, id: int, shift: int, size: int, slope: float):
        self.id: int = id
        self.shift: int = shift
        self.size: int = size
        self.slope: float = slope
    def __lt__(self, other) -> bool:
        return (self.slope < other.slope) if (self.size == other.size) else (self.size < other.size)
    def __gt__(self, other) -> bool:
        return (self.slope > other.slope) if (self.size == other.size) else (self.size > other.size)

class LinearDomainFinder:
    """
    The goal of this class is to act as a simple interface to algorithms for finding
    the Largest Linear Domain (LLD) within a particular data set.

    @@ data members:
    LLD: Domain() - the largest linear domain discovered by the searching algorithm. Stored as a Domain class object.
    popt: np.ndarray - the optimized params as returned by the weighted or unweighted fit of scipy.optimize.curve_fit
    perr: np.ndarray - the error on optimized params as the sqrt of the diagonalization of the covariance matrix returned by curve_fit
    """
    def __init__(self):
        self.LLD: Domain = Domain
        self.popt: np.ndarray = 0.0
        self.perr: np.ndarray = None
        self._domains = None
        self._xdata = None          
        self._xlabel = ""           
        self._ydata = None  
        self._yerr = None        
        self._yabel = ""            
        self._verbosity = 1
        self._method = 0
        self._slopes = []
        self._slope_errors = []

    def setMethod(self, method: int):
        """
        Sets the preferred method for choosing a new domain. Either using FDEV_CUT parameter (0) or
        by direction change (1). FDEV_CUT specifies an error on the domain slope. Direction change only specifies
        a new domain when the slope changes sign.
        """
        self._method = method

    def setX(self, data: list, label="x") -> None:
        """
        Sets the x values of the data points (horizontal axis).
        Also allows the user to pass an optional string as a label for the axis.
        """
        self._xdata = np.array(data)
        self._xlabel = label
    
    def setY(self, data: list, label="y") -> None:
        """
        Sets the y values of the data points (vertical axis).
        Also allows the user to pass an optional string as a label for the axis.
        """
        self._ydata = np.array(data)
        self._ylabel = label

    def setYerr(self, data: list) -> None:
        """
        Sets the yerr array (vertical axis) for curve_fit to perform a weighted fit.
        Otherwise, yerr=None.
        """
        self._yerr = np.array(data)

    def setVerbosity(self, verbose=1) -> None:
        """
        (0 = no output) 
        (1 = some output, only important plots) 
        (2 = all available output and plots, mostly for debugging)
        Default verbosity is 1.
        """
        self._verbosity = verbose
    
    def slidingWindowFind(self, WIN_SIZE: int, FDEV_CUT: float) -> None:
        """
        Sliding Window algorithm which detects different "domains" within the data based on areas where
        the slope is similar. Sliding window size WIN_SIZE (number of points included per regression instance) is an adjustable
        parameter, as well as the fractional deviation (FDEV) of the slope, which effectively sets the algorithm
        sensitivity to variations. A lower FDEV corresponds to higher sensitivity, in which the algorithm
        is likely to find more domains in the data, which may or may not be preferable dependent on the data set.

        If there is a region in which the slope does not change over a significant range of window shifts, it will be identified
        as a region of linear correlation.

        @@ params:
            WIN_SIZE: int - size of the window function, sets the number of data points included in each regression.
            FDEV_CUT: float - fractional deviation allowed in the slope before a new domain is recognized.
        @@ returns:
            No return values. Results are stored in data members of the class
            self.LLD: Domain() - domain object storing information about the discovered largest linear domain
            self.popt: np.ndarray - final parameters of the linear fit
        """
        N = len(self._xdata) - WIN_SIZE + 1     # maximum allowed shift (exclusive) so that the window does not exceed data bounds  
        slopes = np.zeros(N, dtype=float)

        # FIND SLOPE FROM REGRESSION FOR EACH WINDOW SHIFT
        for i in range(N):
            WIN_START = i
            WIN_END = WIN_START + WIN_SIZE

            # perform linear regression on windowed data ("windowed" meaning using list slicing)
            popt = np.polyfit(x=self._xdata[WIN_START:WIN_END], y=self._ydata[WIN_START:WIN_END], deg=1)
            slopes[i] = popt[0]

            if self._verbosity > 1:
                fit_ydata = line(self._xdata[WIN_START:WIN_END], popt[0], popt[1])
                print("Slope:", popt[0])
                plt.title(f"{self._ylabel} vs. {self._xlabel}: Windowing Iteration {i}"); plt.xlabel(self._xlabel); plt.ylabel(self._ylabel)
                if self._yerr is not None:
                    plt.errorbar(self._xdata, self._ydata, yerr=self._yerr, fmt='+', capsize = 2)
                else:
                    plt.plot(self._xdata, self._ydata)
                plt.plot(self._xdata[WIN_START:WIN_END], fit_ydata, 'r-')
                plt.show()

        # FIND ALL DOMAINS BASED ON DEVIATION IN SLOPE
        currSlope = slopes[0]
        currDomainID = 0
        domainIDs = np.zeros(N, dtype=int)  # mostly used for debugging, but also useful for plotting points with color = domainID
        domainIDs[0] = currDomainID
        self._domains = []
        currDomain = Domain(id=currDomainID, shift=0, size=1, slope=currSlope)

        for i in range(1, N):
            # fractional deviation between the current domain's slope and the next slope in the list
            if self._method == 0:
                slope_FDEV = abs(abs(slopes[i] - currSlope) / currSlope)
                if slope_FDEV < FDEV_CUT:
                    currDomain.size += 1
                else:   # store prev domain, update slope, update domain ID and create new domain with updated vals
                    self._domains.append(currDomain)
                    currSlope = slopes[i] 
                    currDomainID += 1          
                    currDomain = Domain(id=currDomainID, shift=i, size=1, slope=currSlope)
                domainIDs[i] = currDomainID
            elif self._method == 1:
                if currSlope * slopes[i] > 0.0:
                    currDomain.size += 1
                else:   # store prev domain, update slope, update domain ID and create new domain with updated vals
                    self._domains.append(currDomain)
                    currSlope = slopes[i] 
                    currDomainID += 1          
                    currDomain = Domain(id=currDomainID, shift=i, size=1, slope=currSlope)
                domainIDs[i] = currDomainID
            
        self._domains.append(currDomain)
        
        if self._verbosity > 0:
            # plot slope against window shift. Linear domains will look like a flat line
            plt.scatter(np.arange(N, dtype=int), slopes, c=domainIDs, cmap='plasma')
            plt.title("Slope vs. Window Shift"); plt.xlabel("Window Shift"); plt.ylabel("Slope"); plt.colorbar()
            plt.show()
        
        # FIND THE LARGEST LINEAR DOMAIN (LLD)
        LLD = self._domains[0]
        for i in range(len(self._domains)):
            if LLD < self._domains[i]:
                LLD = self._domains[i]
        self.LLD = LLD
        
        # USE ALL POINTS BELONGING TO LLD TO FIND ACCURATE SLOPE (use scipy.optimize.curve_fit now instead of np.polyfit)
        for LLD in self._domains:

            LLD_START = LLD.shift
            LLD_END = LLD.shift + LLD.size + WIN_SIZE
            popt = None
            pcov = None
            print(self._yerr)
            if self._yerr is not None:
                popt, pcov = curve_fit(line, xdata=self._xdata[LLD_START:LLD_END], ydata=self._ydata[LLD_START:LLD_END], sigma=self._yerr[LLD_START:LLD_END])
            else:
                popt, pcov = curve_fit(line, xdata=self._xdata[LLD_START:LLD_END], ydata=self._ydata[LLD_START:LLD_END])
            self.popt = popt
            self.perr = np.sqrt(np.diag(pcov))
            self._slopes.append(popt[0])
            self._slope_errors.append(self.perr[1])

            if self._verbosity > 0:
                finalFit_ydata = line(self._xdata[LLD_START:LLD_END], popt[0], popt[1])
                plt.title(f"{self._ylabel} vs. {self._xlabel}: Final Fit (Slope = {self.popt[0]})"); plt.xlabel(self._xlabel); plt.ylabel(self._ylabel)
                if self._yerr is not None:
                    plt.errorbar(self._xdata, self._ydata, yerr=self._yerr, fmt='+', capsize=2)
                else:
                    plt.plot(self._xdata, self._ydata)
                plt.plot(self._xdata[LLD_START:LLD_END], finalFit_ydata, 'r-')
                plt.show()
                if self._verbosity > 1:
                    print("Domain IDs:", domainIDs)


            

        


            

            







    