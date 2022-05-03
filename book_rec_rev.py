from functions import download, dataset_corrections, book_compute, find_author, load_dataset_root, merge_datasets_onISBN, dataset_personalize, prep_dataset_for_correlation, user_input 
import pandas as pd
import numpy as np
import requests
import re

#Please note and check the following:
#
#  * The Python version is: Python3.9 from "C:\Users\frara\anaconda3\python.exe"
#  * The NumPy version is: "1.20.3"


url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'  # optionally webscrapper use possilble
ratings_path = 'BX-Book-Ratings.csv'
book_path = 'BX-Books.csv'
ratings_treshold = 8
# favourite_book = 'the fellowship of the ring (the lord of the rings, part 1)'
# author_of_fav_book = 'tolkien'

# download data file
downl_dataset = download(url)
df_books = downl_dataset['books']
df_ratings = downl_dataset['ratings']

#  optional load rating, books from root
if not downl_dataset: 
    downl_dataset = load_dataset_root(ratings_path, book_path)
    df_books = downl_dataset[0]
    df_ratings = downl_dataset[1]

# ...dataset corections ???
df_books, df_ratings = dataset_corrections(df_books, df_ratings)

# merge books and ratings on ISBN
dataset = merge_datasets_onISBN(df_books, df_ratings)   


# user input
while not author_of_fav_book: 
    favourite_book = user_input()
    fav_book_list = [favourite_book] # correlation works with list, TBD 

    # find author
    author_of_fav_book = find_author(favourite_book, downl_dataset['books'])
    if not author_of_fav_book:
        print(f"Book {favourite_book} or it's author not found in tdb. Please type again.")


# modify to dataset according to author of favourite book
books_of_selected_author_readers = dataset_personalize(dataset, favourite_book, author_of_fav_book)

# prepare dataset for correlations
dataset_for_corr, ratings_data_raw = prep_dataset_for_correlation(books_of_selected_author_readers, ratings_treshold)

# compute correlation
top10results = book_compute(fav_book_list, dataset_for_corr, ratings_data_raw)

# print or send results
# TBD 
print(f'for book: {favourite_book} are 10 most recomended books:\n{top10results}')


#print("Average rating of LOR:", ratings_data_raw[ratings_data_raw['Book-Title']=='the fellowship of the ring (the lord of the rings, part 1'].groupby(ratings_data_raw['Book-Title']).mean()))
# rslt = result_list[0]