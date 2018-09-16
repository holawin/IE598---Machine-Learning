try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import sys
#read data from uci data repository
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urllib2.urlopen(target_url)
#arrange data into list for labels and list of lists for attributes
xList = []
labels = []
for line in data:
    #split on comma
    row = line.strip().split(b",")
    xList.append(row)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])) + '\n')

print("My name is Habeeb Olawin")
print("My NetID is: holawin2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
