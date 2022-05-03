# import
from logging import exception
import zipfile
import pandas as pd
import numpy as np
import requests
import re
from zipfile import ZipFile
from io import BytesIO


def download(url):
    """
    download daat frames from given url
    """
    # TBD any security check...???
    
    def getFilename_fromCd(cd):
        """
        Get filename from content-disposition
        """
        if not cd:
            return None
        
        fname = re.findall('filename=(.+)', cd)
        if len(fname) == 0:
            return None
        
        return fname[0]


    # https://docs.python-requests.org/en/master/user/quickstart/#raw-response-content
    try: 
        r = requests.get(url)       # (opt. param 'stream = true')
    except:
        print(f'{exception}, Dataset not loaded from link. Using version o disk')

    # GET FILENAME
    # if url.find('/'):           
    #     fname=(url.rsplit('/', 1)[1])
    # else:
    #    # user input | default value
    
    # GET FILENAME 2
    # try:
    #    filename=(url.rsplit('/', 1)[1]) 
    # except:
    #    print('Invalid url. Backslash not found.')

    # FILE NAME FROM CONTENT-DISPOSITION HEADER
    # filename = getFilename_fromCd(r.headers.get('content-disposition'))
    
    # WRITE FILE FROM URL
    # with open(filename, 'wb') as f:
    #    for chunk in r.iter_content(chunk_size=128):
    #        f.write(chunk)
    
    my_df_dict = {'ratings' : '', 'books' : ''}
    # open zipped dataset from url
    try:
        with zipfile.ZipFile(BytesIO(r.content)) as my_zip_file:
            file_names = my_zip_file.namelist().sort()
                        
            for contained_file in file_names:
                if 'Ratings' in str(contained_file).split('-')[-1].split('.')[0]:  # ?? str() maybe
                    # open the csv file in the dataset
                    with my_zip_file.open(contained_file) as my_file:                        
                        # read the dataset, 
                        my_df_dict['ratings'] = pd.read_csv(my_file, encoding='cp1251', sep=';',on_bad_lines= 'warn', low_memory=False)
                
                elif 'Books' in str(contained_file).split('-')[-1].split('.')[0]:
                    # open the csv file in the dataset
                    with my_zip_file.open(contained_file) as my_file:                        
                        # read the dataset
                        my_df_dict['books'] = pd.read_csv(my_file, encoding='cp1251', sep=';',on_bad_lines= 'warn', low_memory=False)
                
                else:
                    # other files doesn't need to read
                    continue

    except:
        print(f'{exception}, Dataset not loaded from link. Using version o disk')
        
    return my_df_dict


def load_dataset_root(ratings_p, book_p):
    """
    load ratings, load books from root     
    """    
    # load ratings
    ratings = pd.read_csv(ratings_p, encoding='cp1251', sep=';', on_bad_lines= 'warn')
    ratings = ratings[ratings['Book-Rating']!=0]

    # load books
    books = pd.read_csv(book_p, encoding='cp1251', sep=';', on_bad_lines= 'warn', low_memory=False)   # 'error_bad_lines=False' deprecated, will be removed, dtype = {''}

    return [books, ratings]


def merge_datasets_onISBN(df_books, df_ratings):
    """
    megre of df_books and df_rating on ISBN
    """    
    dataset = pd.merge(df_books, df_ratings, on=['ISBN'])
    
    return dataset


def user_input():
    """
    ask for typing favourite book
    """
    # check, if name exists in dataset, and return match as list of options
    # text formating (e.g. nordic languages letters)?
    # language set?
    
    fav_book_list = [] # future possiblity
    # while ...... not needed yet for 1 book
    favourite_book = input('Type your favourite book.')   

    return favourite_book


def find_author(book, df_books):
    """
    finds author of passed favorite book
    """
    author = df_books['Book-Author'][df_books['Book-Title'].str.lower() == str(book).lower()]
    
    return author


def dataset_corrections(df_books, df_ratings):
    # in general
    # dataset corections in many steps in func    
    # DtypeWarning: Columns (3) have mixed types. Specify dtype option on import or set low_memory=False.
    # dataset lowercase...??? Beware of working with lower() in func, and with object type.
    # .str.strip(), .str.contains('')
    # TBD
    
    df_books = df_books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    df_books = df_books.drop_duplicates(ignore_index = True)    

    df_books['ISBN', 'Book-Title', 'Book-Author', 'Publisher'] = df_books['ISBN', 'Book-Title', 'Book-Author', 'Publisher'].astype('string')
    df_books['Year-Of-Publication'] = df_books['Year-Of-Publication'].astype('int16')  # 2 rows moved columns 
    # my_cond = df_books['Year-Of-Publication'] == 'DK Publishing Inc'
    # help_variable = df_books[my_cond]['Year-Of-Publication'].copy()
    # df_books[my_cond]['Publisher'] = help_variable 

    df_books = df_books.apply(lambda x: x.str.strip() if x.dtype == "string" else x) # trim whitespaces
 
    df_ratings = df_ratings.drop_duplicates(ignore_index = True)
    
    return [df_books, df_ratings]
   


def dataset_personalize(dataset, book_title, author):
    """
    modify and returns dataset acording selected book and author
    """

    dataset_lowercase=dataset.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x) # str as atribut of object type ???

    selected_author_readers = dataset_lowercase['User-ID'][(dataset_lowercase['Book-Title']==book_title) & (dataset_lowercase['Book-Author'].str.contains(author))]
    selected_author_readers = selected_author_readers.tolist()
    selected_author_readers = np.unique(selected_author_readers)

    # user defined dataset
    books_of_selected_author_readers = dataset_lowercase[(dataset_lowercase['User-ID'].isin(selected_author_readers))]

    
    return books_of_selected_author_readers


def prep_dataset_for_correlation(books_of_selected_author_readers, ratings_treshold = 8):
    """
    returns dataset_for_corr, ratings_data_raw
    """
    # Number of ratings per other books in dataset
    number_of_rating_per_book = books_of_selected_author_readers.groupby(['Book-Title']).agg('count').reset_index()

    #select only books which have actually higher number of ratings than threshold
    books_to_compare = number_of_rating_per_book['Book-Title'][number_of_rating_per_book['User-ID'] >= ratings_treshold]
    books_to_compare = books_to_compare.tolist()

    ratings_data_raw = books_of_selected_author_readers[['User-ID', 'Book-Rating', 'Book-Title']][books_of_selected_author_readers['Book-Title'].isin(books_to_compare)]

    # group by User and Book and compute mean
    ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

    # reset index to see User-ID in every row
    ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

    dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

    
    return dataset_for_corr, ratings_data_raw


def book_compute(fav_book_list, dataset_for_corr, ratings_data_raw):
    """
    computes list of top 10 books with highest and bottom 10 books with lowest correlations to each books from argumets list
    """
    result_list = []
    worst_list = []

    # for each of the book of book list compute:
    for book in fav_book_list:
    
        #Take out the Lord of the Rings selected book from correlation dataframe
        dataset_of_other_books = dataset_for_corr.copy(deep=False)
        dataset_of_other_books.drop([book], axis=1, inplace=True)
      
        # empty lists
        book_titles = []
        correlations = []
        avgrating = []

        # corr computation
        for book_title in list(dataset_of_other_books.columns.values):
            book_titles.append(book_title)
            correlations.append(dataset_for_corr[book].corr(dataset_of_other_books[book_title]))
            tab=(ratings_data_raw[ratings_data_raw['Book-Title']==book_title].groupby(ratings_data_raw['Book-Title']).mean())
            avgrating.append(tab['Book-Rating'].min())

        # final dataframe of all correlation of each book   
        corr_fellowship = pd.DataFrame(list(zip(book_titles, correlations, avgrating)), columns=['book','corr','avg_rating'])
        corr_fellowship.head()

        # top 10 books with highest corr
        result_list.append(corr_fellowship.sort_values('corr', ascending = False).head(10))
        
        #worst 10 books
        worst_list.append(corr_fellowship.sort_values('corr', ascending = False).tail(10))

        
    return result_list


