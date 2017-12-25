from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt


corpus = [];
ratings= [];
user_profile = [];
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english');
tweet_matrix=[];
def init_sim(corpus,ratings):
	corpus = corpus;
	ratings = ratings;
	get_user_profile();

def cos_sim():
	return tweet_matrix*np.transpose(user_profile[0])

def get_user_profile():
	global user_profile
	global tweet_matrix
	tfidf_matrix =  tf.fit_transform(corpus);
	user_profile =  tfidf_matrix.multiply(ratings[:, np.newaxis]).sum(axis=0);
	norm = np.linalg.norm(user_profile);
	user_profile = user_profile/norm;
	tfidf_matrix = tfidf_matrix.todense();
	tweet_matrix = tfidf_matrix;

def get_tf_idf_matrix():
		return tweet_matrix;

def nmf_kl_multiplicative(D, M, W, H, EPOCH=20):

    MD = D.copy()

    MD[M==0] = 0

    for e in range(EPOCH):
        Xhat = W.dot(H)
        W*np.array(((MD/Xhat).dot(H.T)/np.dot(M, H.T)))
        Xhat = W.dot(H)
        H = H*np.array((W.T.dot(MD/Xhat)/np.dot(W.T, M)))
        print(np.sum(np.abs(MD - M*Xhat))/np.sum(M))

    return W, H



#Example Usage
corpus=[]
#Tweets
user1_tweets = ("computer architecture and cpu design and embedded systems computer architecture and cpu design and embedded systems computer architecture and cpu design and embedded systems"); # String of last 100 tweets
user2_tweets = ("rock n roll music");
user3_tweets = ("deep learning and neural networks");
user4_tweets = ("age of technology");
user5_tweets = ("political science");
user6_tweets = ("embedded systems and internet of things");
user7_tweets = ("embbeded design"); # String of last 100 tweets
user8_tweets = ("jazz music");
user9_tweets = ("convolutional neural networks");
user10_tweets = ("developments in technology");
user11_tweets = ("political debate");
user12_tweets = ("embedded system design");
user13_tweets = ("computer architecture and cpu design and embedded systems"); # String of last 100 tweets
user14_tweets = ("rock n roll music");
user15_tweets = ("deep learning and neural networks");
user16_tweets = ("age of technology");
user17_tweets = ("political science");
user18_tweets = ("embedded systems and internet of things");
user19_tweets = ("embbeded design"); # String of last 100 tweets
user20_tweets = ("jazz music");
user21_tweets = ("convolutional neural networks");
user22_tweets = ("developments in technology");
user23_tweets = ("political debate");
user24_tweets = ("political debates relating internet and technology");

#Generate Corpus
corpus.append(user1_tweets);
corpus.append(user2_tweets);
corpus.append(user3_tweets);
corpus.append(user4_tweets);
corpus.append(user5_tweets);
corpus.append(user6_tweets);
corpus.append(user7_tweets);
corpus.append(user8_tweets);
corpus.append(user9_tweets);
corpus.append(user10_tweets);
corpus.append(user11_tweets);
corpus.append(user12_tweets);
corpus.append(user13_tweets);
corpus.append(user14_tweets);
corpus.append(user15_tweets);
corpus.append(user16_tweets);
corpus.append(user17_tweets);
corpus.append(user18_tweets);
corpus.append(user19_tweets);
corpus.append(user20_tweets);
corpus.append(user21_tweets);
corpus.append(user22_tweets);
corpus.append(user23_tweets);
corpus.append(user24_tweets);


#Ratings for each user
# 0 for unrated profiles
ratings = np.array([5,1,3,3,1,5,5,1,3,3,1,5,5,1,3,3,1,5,5,1,3,3,1,0]);
#Initialize
init_sim(corpus,ratings);
# Compute similarities
print(cos_sim());

#Rank
R = 1
ratings = np.array([5,1,4,3,1,5,5,1,3,3,1,5,5,1,4,3,1,5,5,1,3,3,1,np.nan]);
tweet_matrix = np.array(tweet_matrix).T;
rating_matrix=ratings.reshape(1,len(ratings));
tweet_matrix[tweet_matrix>0]=1
tweet_matrix = np.append(tweet_matrix, rating_matrix, axis=0)

# Data
Nr = tweet_matrix.shape[0]
Nc = tweet_matrix.shape[1]

# Initialize
W = np.random.rand(Nr, R)*100
H = np.random.rand(R, Nc)*100
Mask = np.ones_like(tweet_matrix)
Mask[np.isnan(tweet_matrix)] = 0

print()
W,H = nmf_kl_multiplicative(tweet_matrix, Mask, W, H, EPOCH=10)
Xhat = W.dot(H)

def ShowMatrix(X, title=''):
    plt.figure()
    plt.imshow(X, interpolation='nearest',vmax=5,vmin=0)
    plt.colorbar()
    plt.set_cmap('jet')
    plt.xlabel('profiles')
    plt.ylabel('features')
    plt.title(title)
    plt.show()

ShowMatrix(tweet_matrix, 'original')
ShowMatrix(Xhat, 'estimate')

print("ratings:")
print(ratings)
print("W")
print(W)
H=np.around(H*5/np.max(H))
print("H")
print(H)

#print(np.argmax(H,axis=0)+1)
