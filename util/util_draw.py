import six
import matplotlib
from matplotlib import colors
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig  
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
from sklearn import metrics
import numpy as np
import copy
import math
import os

def label2cluster(label):
    if type(label)!=list:
        if label.min()!=0:
            label=label-1
    clus=[[] for i in range(len(set(label)))]
    for i,data in enumerate(label):
        clus[data].append(i)
    return clus

def plotFigureLabel(X,label,filename):
   figX=X[:,0]
   figY=X[:,1]
   cluster=label2cluster(label.astype(int))
   color=['Red','Green','Blue','Yellow','Black','Purple','Aqua','Fuchsia','Gray','Lime','Maroon','Navy','Olive','Silver','Teal']
   plt.figure()
   for i in range(len(cluster)):
       plt.scatter(figX[cluster[i]],figY[cluster[i]],c=color[i] if i<len(color) else np.random.rand(3))
   savefig(filename+'.eps')
   savefig(filename+'.png')


def plotFigure(X,cluster,filename):
   figX=X[:,0]
   figY=X[:,1]
   # color = list(six.iteritems(colors.cnames))
   # for name, rgb in six.iteritems(colors.ColorConverter.colors):
       # hex_ = colors.rgb2hex(rgb)
       # color.append((name, hex_))
   color=['Red','Green','Blue','Yellow','Black','Purple','Aqua','Fuchsia','Gray','Lime','Maroon','Navy','Olive','Silver','Teal']
   fig=plt.figure()
   # fig=plt.figure(figsize=(12,6))
   # plt.axis([-7, 12, -5, 5])
   if X.shape[1]==2:
       for i in range(len(cluster)):
           plt.scatter(figX[cluster[i]],figY[cluster[i]],c=color[i] if i<len(color) else np.random.rand(3))
   elif X.shape[1]==3:
       figZ=X[:,2]
       ax = fig.add_subplot(111, projection='3d')
       for i in range(len(cluster)):
           ax.plot(figX[cluster[i]],figY[cluster[i]],figZ[cluster[i]],c=color[i] if i<len(color) else np.random.rand(3))

   fig.savefig(filename+'.eps')
   fig.savefig(filename + '.pdf')
   #fig.savefig(filename+'.png')
   plt.show()
def plotFigure1(X,cluster,filename):
   font = {
           'weight' : 'normal',
           'size'   : 24}

   matplotlib.rc('font', **font)
   figX=X[:,0]
   figY=X[:,1]
   color=['Red','Green','Blue','Yellow','Black','Purple','Aqua','Fuchsia','Gray','Lime','Maroon','Navy','Olive','Silver','Teal']
   fig=plt.figure(figsize = (14, 6))
   plt.axis([-7, 12, -5, 5])

   # plt.axis([-7, 12, -7, 7])
   if X.shape[1]==2:
       for i in range(len(cluster)):
           # plt.scatter(figX[cluster[i]],figY[cluster[i]],s = 60, c=color[i] if i<len(color) else np.random.rand(3))
           plt.plot(figX[cluster[i]],figY[cluster[i]],marker = 'o',linestyle = 'None',markersize = 9,   c=color[i] if i<len(color) else np.random.rand(3))

   fig.savefig(filename+'.eps')
   fig.savefig(filename+'.png')

def plotFigure_outlierColorDiff(X,cluster,outliers,filename):
   figX=X[:,0]
   figY=X[:,1]

   for outlier in outliers:
       cluster.append([outlier])
   color=['Red','Green','Blue','Yellow','Black','Purple','Aqua','Fuchsia','Gray','Lime','Maroon','Navy','Olive','Silver','Teal']
   fig=plt.figure()
   # fig=plt.figure(figsize=(12,6))
   # plt.axis([-7, 12, -5, 5])
   if X.shape[1]==2:
       for i in range(len(cluster)):
           plt.scatter(figX[cluster[i]],figY[cluster[i]],c=color[i] if i<len(color) else np.random.rand(3))
   elif X.shape[1]==3:
       figZ=X[:,2]
       ax = fig.add_subplot(111, projection='3d')
       for i in range(len(cluster)):
           ax.plot(figX[cluster[i]],figY[cluster[i]],figZ[cluster[i]],c=color[i] if i<len(color) else np.random.rand(3))

   fig.savefig(filename+'.eps')
   fig.savefig(filename+'.png')

def Distance(x,y,dist='EP'):
    if dist=='EP':
        return 1-min(min(a,b)/max(a,b) for (a,b) in zip(x,y))
    elif dist=='EQC':
        return math.sqrt(sum([(a-b)**2 for (a,b) in zip(x,y)]))
    elif dist=='CB':
        return max([abs(a-b) for (a,b) in zip(x,y)])
    elif dist=='MH':
        return sum([abs(a-b) for (a,b) in zip(x,y)])
    elif dist=='CON':
        return 1-(np.dot(x, y) / (math.sqrt(np.dot(x, x)) * math.sqrt(np.dot(y, y))))
    elif dist=='PEAR':
        return 1-pearsonr(x,y)[0]

def getScore(y_true,y_pred):
    # pdb.set_trace()
    ARI=adjusted_rand_score(y_true,y_pred)
    NMI = normalized_mutual_info_score(y_true,y_pred)
    PRFS=metrics.classification_report(y_true,y_pred)
    # prec=metrics.precision_score(y_true,y_pred)
    # recall=metrics.recall_score(y_true,y_pred)
    return ARI,NMI

def cluster2label(cluster):
    label={}
    for i,cl in enumerate(cluster):
        for e in cl:
            label[e]=i
    label=label.values()
    # if 'temp' in os.listdir('.'):
      # pass
    # else:
      # os.mkdir('temp')
    # f=open('temp/'+filename+'.label','w')
    # f.write(str(np.ndarray.tolist(y))+'\n')
    # f.write(str(str(label)))
    # f.close()
    return label

def labelNorm(cluster):
    medoids=list(set(cluster))
    medoids.sort(key=cluster.index)
    for index,element in enumerate(cluster):
      for i,me in enumerate(medoids):
        if element!=me:
          pass
        else:
          cluster[index]=i
    return cluster

def cluster2readable(clusters):
    if clusters!=[]:
        if type(clusters[0]) is not list:
            medoids=set(clusters)
            clus=[[] for i in range(len(medoids))]
            for index,d in enumerate(clusters):
                for i,medoid in enumerate(medoids):
                    if d!=medoid:
                        pass
                    else:
                        clus[i].append(index)
            clusterStr=str(clus)
        else:
            clusterStr=str(clusters)
        clusterStr=clusterStr.replace("], ","\n")
        clusterStr=clusterStr.replace("[","")
        clusterStr=clusterStr.replace("]","")
        return clusterStr
    else:
        return '[]'


def find_in_list(mylist,value):
    """If mylist contains "value", return the first position of "value".Else if there are no "value" in mylist ,return -1"""
    pos=-1
    for v in range(0,len(mylist)):
        try:
            if type(mylist[v].index(value)) is int:
                pos=v
                break;
        except:
            pass
    return pos

def mergeCluster(cluster):
    """Cluster is a 2-dimension list,if two sublist of the cluster has the same element,this function will merge these two sublist into one list. Return the new merged cluster list"""
    clu=[]
    for i in range(0,len(cluster)):
        for point in cluster[i]:
            ind=find_in_list(clu,point)
            if ind != -1:
                clu[ind].extend(cluster[i])
                break;
            else:
                pass
        else:
            clu.append(cluster[i])
    return clu

def mergeCluster1(cluster):
    clu=[]
    for i in range(0,len(cluster)):
        merged=-1
        for index,cl in enumerate(clu):
            if (set(cl) & set(cluster[i]))!=set():
              if merged==-1:
                clu[index].extend(cluster[i])
                merged=index
              else:
                clu[merged].extend(clu[index])
                del clu[index]
        else:
          if merged==-1:
            clu.append(cluster[i])
    return clu

def simplifyCluster(cluster):
    """Cluster is a list structure. If a sublist of the list cluster has more than two same elements, then only one will be kept"""
    clu=[]
    for i in range(len(cluster)):
        clu.append(list(set(cluster[i])))
        clu.sort()
    return clu

def duiqi(truth,pres):
    flagp=[-1 for i in range(len(pres))]
    pos=[-1 for i in range(len(truth))]
    yichu=len(pres)+1
    print(truth)
    print(pres)
    for comm in truth:
        print(comm)
        curinmaps={i:(len(list(set(pres[i]).intersection(set(comm))))-len(list(set(pres[i]).difference(set(comm))))*0.00000001) for i in range(len(pres))}
        curinmaps=sorted(curinmaps.items(), key=lambda curinmaps : curinmaps[1]) 
        print(curinmaps)
        flag=0
        maxcurr=copy.deepcopy(curinmaps.pop())
        maxpre=copy.deepcopy(pres[maxcurr[0]])
        while isCorrect(truth,pres,flagp,maxpre,pos,comm)==1 or maxcurr[1]<=0:
            if len(curinmaps)==0:
                posi=truth.index(comm)
                pos[posi]=yichu
                yichu=yichu+1
                flag=1
                break
            maxcurr=copy.deepcopy(curinmaps.pop())
            maxpre=copy.deepcopy(pres[maxcurr[0]])
        if flag==0:
            flagp[maxcurr[0]]=1
            posi=truth.index(comm)
            pos[posi]=maxcurr[0]
    return pos
def isCorrect(truth,pres,flagp,maxpre,pos,comm):
    posp=pres.index(maxpre)
    if flagp[posp]==1:
        return 1
    else:
        commpos=truth.index(comm)
        premaps={}
        for i in range(len(truth)):
            if pos[i]==-1:
                premaps[i]=len(list(set(truth[i]).intersection(set(maxpre))))-len(list(set(truth[i]).difference(set(maxpre))))*0.00000001
        premaps=sorted(premaps.items(), key=lambda premaps : premaps[1],reverse=True) 
        if premaps[0][0]==commpos and premaps[0][1]>0:
            return 0
        else:
            return 1
