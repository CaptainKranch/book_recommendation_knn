import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.neighbors import  KNeighborsClassifier, NearestNeighbors

PATH = os.getcwd()

books_filename = os.path.join(PATH, 'book-crossings', 'BX-Books.csv')
ratings_filename = os.path.join(PATH, 'book-crossings', 'BX-Book-Ratings.csv')

print(books_filename)

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})



books_head = df_books.head()
ratings_head = df_ratings.head()
print(books_head)
print(ratings_head)

print(df_books.isnull().sum())
print(df_ratings.isnull().sum())
print(df_books.dropna(inplace=True))
df_books.isnull().sum()


# print(df_ratings.shape)
ratings = df_ratings['user'].value_counts() # we have to use the original df_ratings to pass the challenge
trash_ratings = (ratings[ratings < 200])
trash_users = df_ratings['user'].isin(trash_ratings.index)

df_users_rm = df_ratings[
  ~trash_users
]
# print(df_users_rm.shape)

ratings = df_ratings['isbn'].value_counts()
df_ratings_rm = df_users_rm[
  ~df_users_rm['isbn'].isin(ratings[ratings < 100].index)
]
# print(df_ratings_rm.shape)

books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
        "I'll Be Seeing You",
        "The Weight of Water",
        "The Surgeon",
        "I Know This Much Is True"]

for book in books:
  df_ratings_rm.isbn.isin(df_books[df_books.title == book].isbn).sum()

df = df_ratings_rm.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T
# print(df.head())

df.index = df.join(df_books.set_index('isbn'))['title']
df = df.sort_index()
print(df.head())

print(df.loc["The Queen of the Damned (Vampire Chronicles (Paperback))"][:10])

model = NearestNeighbors(metric='cosine')
model.fit(df.values)

title = 'The Queen of the Damned (Vampire Chronicles (Paperback))'
distance, indice = model.kneighbors([df.loc[title].values], n_neighbors=6)

print(distance)
print(indice)

print(df.iloc[indice[0]].index.values)


# function to return recommended books - this will be tested
def get_recommends(title = ""):
  try:
    book = df.loc[title]
  except KeyError as e:
    print('The given book', e, 'does not exist')
    return

  distance, indice = model.kneighbors([book.values], n_neighbors=6)

  recommended_books = pd.DataFrame({
      'title'   : df.iloc[indice[0]].index.values,
      'distance': distance[0]
    }) \
    .sort_values(by='distance', ascending=False) \
    .head(5).values

  return [title, recommended_books]

  get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")

  books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You havn't passed yet. Keep trying!")

test_book_recommendation()
