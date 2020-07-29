# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:34:56 2020

@author: saren
"""
import spotipy
import spotipy.util as util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler

import numpy as np

#auth
username = ''
user_id = ''
scope = 'user-top-read'
redirect_uri = 'http://localhost:7777/callback'
limit = 50

client_id = ''
secret = ''

token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=secret, redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)


#top songs of users

def user_top_tracks(limit):
    top_tracks = sp.current_user_top_tracks(limit=limit, time_range = 'short_term')
    top_tracks_df = pd.DataFrame(columns = ['track_id'])
    
    for i in range(limit):
        track_name = top_tracks['items'][i]['name']
        track_id = top_tracks['items'][i]['id']
        track_release = top_tracks['items'][i]['album']['release_date']
        track_pop = top_tracks['items'][i]['popularity']
        track_artist = top_tracks['items'][i]['artists'][0]['name']
        
        audio = sp.audio_features(tracks=track_id)
        top_tracks_df = top_tracks_df.append({'track_name': track_name, 
                                              'artist_name': track_artist,
                                  'release_date': track_release,
                                    'track_id': track_id, 
                                   'popularity': track_pop,
                                  'time_signature': audio[0]['time_signature'],
                                  'key': audio[0]['key'],
                                   'energy': audio[0]['energy'],
                                  'acousticness': audio[0]['acousticness'],
                                  'danceability': audio[0]['danceability'],
                                  'instrumentalness':audio[0]['instrumentalness'],
                                  'liveness':audio[0]['liveness'],
                                  'loudness': audio[0]['loudness'],
                                  'speechiness': audio[0]['speechiness'],
                                  'tempo': audio[0]['tempo'],                  
                                  'valence': audio[0]['valence']},
                                  ignore_index=True)
    return top_tracks_df
    
user_top_tracks = user_top_tracks(limit)


#database of all songs
songs_database = pd.read_csv('../SpotifyFeatures.csv')
all_songs_data = songs_database.append(user_top_tracks)

#correlation
correlation = all_songs_data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(correlation, annot=True)

high_corr = correlation[abs(correlation) >= 0.5]

#clustering
asd_cluster = all_songs_data.copy()
asd_cluster.to_csv('all_songs_database.csv', index=0)

#pre-processing
X = pd.DataFrame(asd_cluster.iloc[:, [4,5,6,8,9,11,12]].values)
cols = asd_cluster.iloc[:, [4,5,6,8,9,11,12]].columns
X.columns = cols

scaler = MinMaxScaler()

scaled = pd.DataFrame(scaler.fit_transform(X))
scaled.columns = cols
scaled= scaler.fit_transform(X)

#machine learning/kmeans
val_per_cluster = []
for i in range(1,1000,50):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled)
    val_per_cluster.append(kmeans.inertia_)
    
plt.plot(range(1,1000,50), val_per_cluster,'o')
plt.plot(range(1,1000,50), val_per_cluster,'-')
plt.ylabel('Number of songs')
plt.xlabel('Number of clusters')

kmeans = KMeans(n_clusters = 850, random_state=0)
y_kmeans = kmeans.fit_predict(scaled)

#creating playlists and adding songs
int_cluster_songs = asd_cluster.tail(50)
int_cluster = y_kmeans[-50:]
int_cluster = np.unique(int_cluster)

def create_playlist(i):
    scope = 'playlist-modify-private'
    token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth=token)

    playlist_name = f'Cluster {i}'
    playlist = sp.user_playlist_create(user=user_id, name = playlist_name, public = False)
    playlist_id = playlist['id']
    return playlist_id

def add_songs_playlist(i, playlist_id):
    scope = 'playlist-modify-private'
    token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=secret, redirect_uri=redirect_uri)
    sp = spotipy.Spotify(auth=token)
    
    tracks_id = asd_cluster[asd_cluster['cluster']==i]['track_id']
    sample = tracks_id.sample(n=20)
    s = list(sample)
    add = sp.user_playlist_add_tracks(user=username, playlist_id=playlist_id, tracks = s)
    return add

for i in int_cluster:   
    playlist_id = create_playlist(i)
    playlist = add_songs_playlist(i, playlist_id)
    
    
