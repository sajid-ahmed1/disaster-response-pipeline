import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    returns: merged dataframe between messages and categories joined left by id
    
    output: 
        df: a Pandas dataframe
    
    input: 
        messages_filepath: csv filepath
        categories_filepath: csv filepath
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')
    return df

def clean_data(df):
    '''
    returns: a cleaned dataframe of df with categories expanded into 36 columns and their values 1 or 0
    
    output: 
        df: cleaned Pandas dataframe
    
    input: 
        df: a Pandas dataframe
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(subset=['message'])
    return df

def save_data(df, database_filename):
    '''
    input:
        df: a Pandas dataframe
        database_filename: the database filepath
        
    output:
        df.to_sql: saves dataframe into SQL database under filepath
    '''
    engine = create_engine('sqlite:///' + database_filename)
    return df.to_sql('df', engine,if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()