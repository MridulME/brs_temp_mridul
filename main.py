from table_reader import connect_to_mysql, refresh_data, start_background_refresh, transform_book_df
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import logging
import threading
import time

app = Flask(__name__)

# Connect to MySQL and load initial data
book_df, user_df, book_query, user_query, engine = connect_to_mysql()

def train_model(book_df):
    try:
        all_titles = book_df['Title'].tolist()
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_titles)
        knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        knn_model.fit(tfidf_matrix)
        return tfidf_vectorizer, knn_model, tfidf_matrix
    except KeyError as e:
        logging.error(f"Error during model training: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None, None, None

# Train the initial model
tfidf_vectorizer, knn_model, tfidf_matrix = train_model(book_df)

def recommend_books(interest_areas, book_df, tfidf_vectorizer, knn_model, similarity_threshold=0.1):
    try:
        interest_areas_list = [ia.strip() for ia in interest_areas.split(',')]
        interest_areas_tfidf = tfidf_vectorizer.transform(interest_areas_list)

        matching_books_indices = set()
        for interest_area_tfidf in interest_areas_tfidf:
            distances, indices = knn_model.kneighbors(interest_area_tfidf, n_neighbors=len(book_df))
            matching_indices = [indices[0][i] for i in range(len(distances[0])) if distances[0][i] <= (1 - similarity_threshold)]
            matching_books_indices.update(matching_indices)

        if matching_books_indices:
            recommended_books = book_df.iloc[list(matching_books_indices)]
            return recommended_books
        else:
            return "No matching books found"
    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        return "No matching books found due to an error."

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        global book_df, user_df, tfidf_vectorizer, knn_model
        # Refresh data before recommending
#        book_df, user_df = refresh_data(engine, book_query, user_query)
#        book_df = transform_book_df(book_df)
        tfidf_vectorizer, knn_model, tfidf_matrix = train_model(book_df)

        user_id = int(user_id)
        user_interest = user_df.loc[user_df['professor_id'] == user_id, 'interest_area'].values[0]
        recommendations = recommend_books(user_interest, book_df, tfidf_vectorizer, knn_model)
        if isinstance(recommendations, pd.DataFrame):
            recommended_books = recommendations.to_dict(orient='records')
            return jsonify(recommended_books), 200
        else:
            return jsonify({"message": recommendations}), 200
    except IndexError:
        return jsonify({"error": f"No user found with ID {user_id}"}), 404
    except ValueError:
        return jsonify({"error": "Invalid user ID"}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route('/recommend_by_interest', methods=['GET'])
def recommend_by_interest():
    interest_area = request.args.get('interest_area')
    if not interest_area:
        return jsonify({"error": "Interest area is required"}), 400

    try:
        global book_df, user_df, tfidf_vectorizer, knn_model
        # Refresh data before recommending
#        book_df, user_df = refresh_data(engine, book_query, user_query)
#        book_df = transform_book_df(book_df)
        tfidf_vectorizer, knn_model, tfidf_matrix = train_model(book_df)

        recommendations = recommend_books(interest_area, book_df, tfidf_vectorizer, knn_model)
        if isinstance(recommendations, pd.DataFrame):
            recommended_books = recommendations.to_dict(orient='records')
            return jsonify(recommended_books), 200
        else:
            return jsonify({"message": recommendations}), 200
    except Exception as e:
        logging.error(f"Error during recommendation: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# Periodically retrain model
def retrain_model_periodically(interval):
    while True:
        time.sleep(interval)
        try:
            global book_df, user_df, tfidf_vectorizer, knn_model, tfidf_matrix
            # Refresh data and retrain the model
            book_df, user_df = refresh_data(engine, book_query, user_query)
            book_df = transform_book_df(book_df)
            tfidf_vectorizer, knn_model, tfidf_matrix = train_model(book_df)
            logging.info("Model retrained successfully.")
        except Exception as e:
            logging.error(f"Error during model retraining: {e}")

# Set retraining interval (e.g., every 1 hour)
retraining_interval = 3600
retraining_thread = threading.Thread(target=retrain_model_periodically, args=(retraining_interval,))
retraining_thread.daemon = True
retraining_thread.start()

if __name__ == '__main__':
    # Start background refresh thread
    start_background_refresh(book_df, user_df, book_query, user_query, engine, interval=3600)
    app.run(host='0.0.0.0', port=5000, debug=True)
