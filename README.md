# automate-spotify-playlists

### Description
Spotify uses sophisticated machine learning algorithms to generate Discover Weekly playlists for users. These Discover Weekly playlists consist of new songs for the listener and is curated based on the user's listening history. This project can not truly replicate or beat Spotify's methods of music curation due to the high complexity of their algorithms. The purpose of this project is to automate the curation of spotify playlist based on a user's top tracks (according to their last month's listening history). This is done through KMeans clustering, a method to group n variables into k clusters. Each observation belong to a cluster with the closest mean. 

### Learning Objectives
1. Learn how to obtain data from the Spotify API and use collected data to create programs through the Spotipy Python package
2. Improve understanding and application of pandas
3. Ultilize sklearn to create a simple classification model
4. Gain an understanding of KMean clustering

### Packages used
1. Spotipy - used to access the Spotify API and perform actions in the Spotify application
2. pandas - import csv file, organize information of songs
3. matplotlib - create plots
4. sklearn - to create Kmean models, pre-process data

### Files in the repository 
- automate_spotify_playlists.py: all of the code for the program
- Automate-Spotify-Playlists.ipynb: Jupyter Notebook version of the py file
- SpotifyFeatures.csv: database of 160k+ songs on Spotify, found on https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks
- elbow_method_spotify.png: image of the elbow means graph
- example.png: image of one of the playlists created
