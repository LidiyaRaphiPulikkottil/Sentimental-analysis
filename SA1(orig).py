from scipy.io.wavfile import read  # to read wavfiles
import matplotlib.pyplot as plotter
from sklearn.tree import DecisionTreeClassifier as dtc
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from sklearn import tree
import pydotplus
import numpy as np
import os
import scipy
import math

fList = []	#feature list
mfList = [] #main feature list
labels = ["angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "angry", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear",  "fear",  "fear",  "fear",  "fear",  "fear", "fear", "fear", "fear", "fear", "fear", "fear", "fear", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "happy", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad"
, "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad", "sad"]	#label set

def stddev(lst,mf):
    sum = 0
    for i in range(len(lst)):
        sum += pow((lst[i]-mf),2)
    sd = np.sqrt(sum/len(lst)-1)
    fList.append(sd)#standard deviation

def find_iqr(num,num_array=[],*args):
	num_array.sort()
	l=int((int(num)+1)/4)
	m=int((int(num)+1)/2)
	med=num_array[m]
	u=int(3*(int(num)+1)/4)
	fList.append(num_array[l])	#first quantile
	fList.append(med)	#median
	fList.append(num_array[u])	#third quantile
	fList.append(num_array[u]-num_array[l])	#inter quantile range


def build(path):
	dirlist=os.listdir(path)
	dirlist.sort()
	n=1
	for name in dirlist:
		global fList
		path3=path+name
		print ("File ",n,name)
		fs, x = read(path3) #fs will have sampling rate and x will have sample #
		#print ("The sampling rate: ",fs)
		#print ("Size: ",x.size)
		#print ("Duration: ",x.size/float(fs),"s")
	
		'''
		plotter.plot(x)
		plotter.show() #x-axis is in samples 
		t = np.arange(x.size)/float(fs) #creating an array with values as time w.r.t samples
		plotter.plot(t)   #plot t w.r.t x
		plotter.show()
		y = x[100:600]
		plotter.plot(y)
		plotter.show()  # showing close-up of samples 
		'''
		j=0
		mf=0
		med=0
		for i in x:
			j=j+1
			mf=mf+i
		mf=mf/j
		fList.append(np.max(abs(x)))	#amplitude
		fList.append(mf)	#mean frequency
		find_iqr(j,x)
		fList.append((3*med)-(2*mf)) 	#mode
		stddev(x,mf)
		#fftc = np.fft.rfft(x).tolist()
		#mr = 20*scipy.log10(scipy.absolute(x)).tolist()
		#fList.append(fftc)	#1D dft
		#fList.append(mr)	#magnitude response
		
		(rate,sig) = read(path3)
		mfcc_feat = mfcc(sig,rate)
		d_mfcc_feat = delta(mfcc_feat, 2)
		fbank_feat = logfbank(sig,rate)
		fl = []
		fl2 = []

		for l1 in mfcc_feat[50:100]:
			for l2 in l1:
				fl.append(l2)
				
		for l1 in fbank_feat[50:100,:]:
			for l2 in l1:
				fl2.append(l2)

		fl = ['%.4f' % elem for elem in fl]
		fl2 = ['%.4f' % elem for elem in fl2]
		
		
		for l1 in fl:
			fList.append(l1)
		for l1 in fl2:
			fList.append(l1)
		
		mfList.append(fList)
		fList = []
		n=n+1
		
		
path1 = '/home/hp/Desktop/Trainingsamplesmono/'
path2 = '/home/hp/Desktop/set9/'
clf = dtc() # this class is used to make decision tree
build(path1)

clf.fit(mfList,labels)
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris1.pdf") 
mfList = []	#clear mflist
build(path2)
res = clf.predict(mfList)	# prediction of sentiments
print(res)
