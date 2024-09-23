import time
import threading
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import config as c
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_mysql():
    """
    Establishes a connection to the MySQL database and loads initial data into DataFrames.
    """
    try:
        db_uri = c.db_uri_conf
        # Set up connection pooling
        engine = create_engine(db_uri, pool_size=10, max_overflow=20)
        
        with engine.connect() as connection:
            book_query = f"SELECT * FROM {c.book_data_for_reader}"
            user_query = """SELECT user_id as professor_id,
                            GROUP_CONCAT(interested_in ORDER BY interested_in ASC SEPARATOR ' ') AS interest_area 
                            FROM user_interest GROUP BY user_id"""

            book_df = pd.read_sql_query(book_query, connection)
            user_df = pd.read_sql_query(user_query, connection)

        # Transform the book_df DataFrame
        book_df = transform_book_df(book_df)

        logging.info("Initial data loaded successfully.")
        return book_df, user_df, book_query, user_query, engine

    except SQLAlchemyError as e:
        logging.error(f"Error connecting to the database: {e}")
        return None, None, None, None, None

def refresh_data(engine, book_query, user_query):
    """
    Fetches the latest data from the database using the existing engine.
    """
    try:
        with engine.connect() as connection:
            book_df = pd.read_sql_query(book_query, connection)
            user_df = pd.read_sql_query(user_query, connection)

        # Transform the book_df DataFrame
        book_df = transform_book_df(book_df)

        return book_df, user_df
    except SQLAlchemyError as e:
        logging.error(f"Error refreshing data from the database: {e}")
        return None, None

def transform_book_df(book_df):
    """
    Applies the necessary transformations to the book_df DataFrame.
    """
    book_df['published_date'] = pd.to_datetime(book_df['published_date'], errors='coerce')
    book_df['created_date'] = pd.to_datetime(book_df['created_date'], errors='coerce')
    book_df['published_date'] = book_df['published_date'].fillna('null')
    book_df['created_date'] = book_df['created_date'].fillna('null')
    book_df.rename(columns={"name": "Title"}, inplace=True)
    return book_df

def start_background_refresh(book_df, user_df, book_query, user_query, engine, interval=3600):
    """
    Continuously checks for updates to the DataFrames at the specified interval.
    """
    def check_for_updates():
        nonlocal book_df, user_df
        while True:
            try:
                new_book_df, new_user_df = refresh_data(engine, book_query, user_query)

                if new_book_df is not None and not new_book_df.equals(book_df):
                    book_df = new_book_df
                    logging.info("Book DataFrame updated")

                if new_user_df is not None and not new_user_df.equals(user_df):
                    user_df = new_user_df
                    logging.info("User DataFrame updated")

                time.sleep(interval)

            except SQLAlchemyError as e:
                logging.error(f"Error checking for updates: {e}")
                time.sleep(interval)

    # Start the background thread to check for updates
    threading.Thread(target=check_for_updates, daemon=True).start()
