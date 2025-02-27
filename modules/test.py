# TEST SCRIPT OF where_linear.py's LinearDomainFinder

from where_linear import LinearDomainFinder

xdata = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
ydata = [0.5, 10.5, 20.5, 30.5, 40.5, 50.5, 62.5, 76.5, 92.5, 110.5]
yerr = [3.0, 2.5, 3.5, 4.5, 2.5, 2.5, 5.5, 6.5, 3.5, 4.5]

LDF = LinearDomainFinder()
LDF.setVerbosity(1)
LDF.setX(xdata, label='time')
LDF.setY(ydata, label='Voltage')
LDF.setYerr(yerr)

LDF.slidingWindowFind(WIN_SIZE=3, FDEV_CUT=0.1)

print(LDF.popt)
print(LDF.perr)


