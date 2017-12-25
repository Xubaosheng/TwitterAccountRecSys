from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import math

italian_stopwords = ['a', 'adesso', 'ai', 'al' ,'alla' , 'allo' , 'allora' , 'altre']
italian_stopwords += ['altri', 'altro', 'anche', 'ancora' ,'avere' , 'aveva' , 'avevano' , 'ben']
italian_stopwords += ['buono', 'che', 'chi', 'cinque' ,'comprare' , 'con' , 'consecutivi' , 'consecutivo']
italian_stopwords += ['cosa', 'cui', 'da', 'del' ,'della' , 'dello' , 'dentro' , 'deve']
italian_stopwords += ['devo', 'di', 'doppio', 'due' ,'e' , 'ecco' , 'fare' , 'fine']
italian_stopwords += ['fino', 'fra', 'gente', 'giu' ,'ha' , 'hai' , 'hanno' , 'ho']
italian_stopwords += ['il', 'indietro', 'invece', 'io' ,'la' , 'lavoro' , 'le' , 'lei']
italian_stopwords += ['lo', 'loro', 'lui', 'lungo' ,'ma' , 'me' , 'meglio' , 'molta']
italian_stopwords += ['molti', 'molto', 'nei', 'nella' ,'no' , 'noi' , 'nome' , 'nostro']
italian_stopwords += ['nove', 'nuovi', 'nuovo', 'o' ,'oltre' , 'ora' , 'otto' , 'peggio']
italian_stopwords += ['pero', 'persone', 'piu', 'poco' ,'primo' , 'promesso' , 'qua' , 'quarto']
italian_stopwords += ['quasi', 'quattro', 'quello', 'questo' ,'qui' , 'quindi' , 'quinto' , 'rispetto']
italian_stopwords += ['sara', 'secondo', 'sei', 'sembra' ,'sembrava' , 'senza' , 'sette' , 'sia']
italian_stopwords += ['siamo', 'siete', 'solo', 'sono' ,'sopra' , 'soprattutto' , 'sotto' , 'stati']
italian_stopwords += ['stato', 'stesso', 'su', 'subito' ,'sul' , 'sulla' , 'tanto' , 'te']
italian_stopwords += ['tempo', 'terzo', 'tra', 'tre' ,'triplo' , 'ultimo' , 'un' , 'una']
italian_stopwords += ['uno', 'va', 'vai', 'voi' ,'volte' , 'vostro']

class TwitterAccountSimilarityFinder:
	corpus = [];
	ratings= [];
	user_profile = [];
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = 'english');
	#tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = italian_stopwords);

	tweet_matrix=[];

	def __init__(self, corpus, ratings):
		self.corpus = corpus;
		self.ratings = np.array(ratings);
		self.get_user_profile()

	def cos_sim(self):
		return self.tweet_matrix*np.transpose(self.user_profile[0])

	def get_user_profile(self):
		global user_profile
		global tweet_matrix

		tfidf_matrix =  self.tf.fit_transform(self.corpus);
		user_profile =  tfidf_matrix.multiply(self.ratings[:, np.newaxis]).sum(axis=0);
		norm = np.linalg.norm(user_profile);
		self.user_profile = user_profile/norm;

		tfidf_matrix = tfidf_matrix.todense();
		self.tweet_matrix = tfidf_matrix;

def main():
	print("Tf-idf vectorizer ran. Example:")
	#Example Usage
	corpus=[]
	#Tweets
	user1_tweets = ("computer architecture and cpu design and embedded systems"); # String of last 100 tweets
	user2_tweets = ("rock n roll music");
	user3_tweets = ("deep learning and neural networks");
	user4_tweets = ("age of technology");
	user5_tweets = ("political science");
	user6_tweets = ("embedded systems and internet of things");

	#Generate Corpus
	corpus.append(user1_tweets);
	corpus.append(user2_tweets);
	corpus.append(user3_tweets);
	corpus.append(user4_tweets);
	corpus.append(user5_tweets);
	corpus.append(user6_tweets);


	#Ratings for each user
	# 0 for unrated profiles
	ratings = np.array([5,2,1,4,1,0]);
	# Compute similarities
	TASF= TwitterAccountSimilarityFinder(corpus,ratings)
	print(TASF.cos_sim());

if __name__ == "__main__":
    main()
