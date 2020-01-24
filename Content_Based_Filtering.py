#First we import things
import pandas as import pd 
from math import sqrt
import numpy as np 
import matplotlib.pyplot as plt  
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
#Then we will process movie file a bit
#Each movie name has yaer inside bracket,which we will extract 
movies_df['year']=movies_df.title.str.extract('(\d(\d\d\d\d\))',expand = False)
movies_df['year']=movies_df.year.str.extract('(\d\d\d\d)',expand = False)
movies_df['title']=movies_df.title.str.replace('(\d(\d\d\d\d\))','')
movies_df['title']=movies_df.title.apply(lambda x:x.strip()) 
#Now let's process genre since every genre is seperated by a '|'
movies_df['genres']=movies_df.genres.str.split('|')
#Now having a list is not optimal so let's make
#a  matrix with binary values with all possible genres
moviesWithGenres_df = movies_df.copy() 
for index,row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index,genre]=1
#Let's fill the Nan values with zeros
moviesWithGenres_df = moviesWithGenres_df.fillna(0) 
#Now look at the ratings info
#Timestamp is useless so drop it
ratings_df = ratings_df.drop('timestamp',1) 
#Now we will process a user's input 
userInput = [
   {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
]
inputMovies = pd.DataFrame(userInput) 
#If we look at the input movies, we actually need the movies ID
#and also their genre matrix to predict some new movie
#Sp we will update iunput movies merging them with genre and movie ID
#Filtering out the movies by title 
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())] 
inputMovies = pd.merge(inputId, inputMovies) 
#Dropping information we don't use in the dataframe
inputMovies.drop('genres',1).drop)('year',1)
#Now we will get the genre matrix from movieswithGenres_df
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())] 
#We will only need the genre table, let's clran it further by dropping 
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop= True)
userGenreTable = userMovies.drop('year',1).drop('movieId',1).drop('genres',1).drop('title',1) 
userProfile = userGenreTable.tranpose().dot(inputMovies['ratings'])
#This is a matrix with points for each genre 
#Now let's get the genres of all movies in our original dataframe to suggest some good movies to 
#this guy
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId',1).drop('year',1).drop('genres',1).drop('title',1) 
recommmendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())  
movies_df.loc[movies_df['movieId'].isin(recommmendationTable_df.head(20))]


