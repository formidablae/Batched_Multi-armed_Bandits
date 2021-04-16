import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def prepare_data(movies,
                 # users,
                 ratings,
                 m):
    print('Inizio preparazione dati...')
    movies.columns = ['MovieID', 'Title', 'Genres']
    # users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Remove second genre from movies categories with multiple genres
    movies['Genres'] = movies['Genres'].str.split('|').str[0]

    print('Inizio sceta top m bandits...')
    # Choosing the bandits the m users with most ratings of movies
    topmBanditsRatings = ratings[ratings['UserID'].isin(ratings['UserID'].value_counts()[:m].index.tolist())]
    print('Fine sceta top m bandits.')

    print('Inizio rinomina bandits ids...')
    topmBanditsRatings['UserID'].replace(topmBanditsRatings['UserID'].unique().tolist(), range(1, m + 1),
                                         inplace=True)
    print('Fine rinomina bandits ids.')

    print('Inizio merging ratings con movie genres...')
    # Merging ratings with movie genras (arms)
    topmBanditsRatings = pd.merge(left=topmBanditsRatings, right=movies,
                                                    left_on='MovieID', right_on='MovieID').drop(columns=['Title'])
    print('Fine merging bandit ratings con movie genres.')

    print('Inizio rinomina genres...')
    # Map Genres to numbers going from 1 to 18,
    topmBanditsRatings['Genres'].replace(['Action', 'Adventure', 'Animation',
                                          'Children\'s', 'Comedy', 'Crime',
                                          'Documentary', 'Drama', 'Fantasy',
                                          'Film - Noir', 'Horror', 'Musical',
                                          'Mystery', 'Romance', 'Sci - Fi',
                                          'Thriller', 'War', 'Western'],
                                         range(1, 19),
                                         inplace=True)
    print('Fine rinomina genres')

    print('Inizio flattening dei ratings...')
    topmBanditsRatings['Rating'].replace(range(1, 6), [1, 1, 1, 2, 2], inplace=True)
    print('Fine flattening ratings.')

    print('Inizio rimozione e rinomina colonne...')
    # Elimino colonne non necessarie, rinomino quelle necessarie
    topmBanditsRatings = topmBanditsRatings.drop(columns=["MovieID"]).rename(
        columns={'UserID': 'Bandit', 'Rating': 'Reward_mu', 'Timestamp': 'Time_T', 'Genres': 'Arm_K'})
    print('Fine rimozione e rinomina colonne.')

    print('Fine preparazione dati.')
    return topmBanditsRatings
