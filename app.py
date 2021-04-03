import flask
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import itertools

app = flask.Flask(__name__, template_folder='templates')

reviews_cleaned = pd.read_csv('./data/reviews_cleaned.csv')
reviews_df_final = pd.read_csv('./data/reviews_df_final.csv')
X = reviews_cleaned['review_cleaned']

# load the model from disk
filename = "./models/XGB_tfidf_smote.sav"
best_model = pickle.load(open(filename, 'rb'))

tfidf_vect = TfidfVectorizer(ngram_range=(1, 1),max_features=5000)
tfidf_vect_X = tfidf_vect.fit_transform(X)
tfidf_vect_X = tfidf_vect_X.toarray()

y_pred = best_model.predict(tfidf_vect_X)
y_prob = best_model.predict_proba(tfidf_vect_X)
reviews_df_final['predicted_sentiment'] = y_pred
product_sentiment_df = pd.DataFrame(reviews_df_final.groupby('name')['predicted_sentiment'].mean()).reset_index()

all_user_names = list(np.unique([*itertools.chain.from_iterable(reviews_df_final.reviews_username)]))


def get_recommendations(user_input):
    #User-user collaborative Filtering
    reviews_recco = reviews_df_final[['reviews_username', 'name', 'reviews_rating']]
    #Using Adjusted Cosine
    # Create a user-product matrix.
    df_pivot = reviews_recco.pivot_table(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    )
    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T - mean).T
    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    user_correlation[user_correlation < 0] = 0
    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    # Copy the train dataset into dummy_train
    dummy = reviews_recco.copy()

    dummy.head()

    # The products not rated by user is marked as 1 for prediction.
    dummy['reviews_rating'] = dummy['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)
    # Convert the dummy train dataset into matrix format.
    dummy = dummy.pivot_table(
        index='reviews_username',
        columns='name',
        values='reviews_rating',
    ).fillna(1)
    user_final_rating = np.multiply(user_predicted_ratings, dummy)
    d = pd.DataFrame(user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]).reset_index()
    final_recco = d.merge(product_sentiment_df, on='name', how='inner')
    final_recco = final_recco.sort_values(['predicted_sentiment'], ascending=False)[0:5]

    return final_recco


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['user_name']
        #m_name = m_name.title()
        result_final = get_recommendations(m_name)
        names = []
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])

        return flask.render_template('positive.html', product_names=names, search_name=m_name)

# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route("/predict", methods=['POST'])
# def predict():
#     if (request.method == 'POST'):
#         int_features = [x for x in request.form.values()]
#         final_features = [np.array(int_features)]
#         output = model_load.predict(final_features).tolist()
#         return flask.render_template('index.html', prediction_text='Churn Output {}'.format(output))
#     else :
#         return render_template('index.html')


if __name__ == '__main__':
    app.run()


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)