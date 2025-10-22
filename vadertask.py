

DATABASE = 'database.sqlite'
def main():
    import sqlite3
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download('vader_lexicon')

    # Connect to the database
    conn = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Fetch all contents
    cursor.execute('SELECT content FROM posts')
    posts = cursor.fetchall()
    cursor.execute('SELECT content FROM comments')
    comments = cursor.fetchall()

    sia = SentimentIntensityAnalyzer()

    # sentiment scores for posts and comments
    post_scores = [sia.polarity_scores(row['content'])['compound'] for row in posts]
    comment_scores = [sia.polarity_scores(row['content'])['compound'] for row in comments]

    # Overall Platform Tone
    all_scores = post_scores + comment_scores
    if len(all_scores) > 0:
        overall_sentiment = sum(all_scores) / len(all_scores)
    else:
        overall_sentiment = 0

    print(f'Overall Platform Sentiment Score: {overall_sentiment:.4f}')

    # These numbers are found from medium.com article. link: https://medium.com/@skillcate/sentiment-analysis-using-nltk-vader-98f67f2e6130
    if overall_sentiment >= 0.05:
        print('The overall tone of the platform is Positive.')

    elif overall_sentiment <= -0.05:
        print('The overall tone of the platform is Negative.')
    else:
        print('The overall tone of the platform is Neutral.')

    # Close the database connection
    conn.close()


if __name__ == '__main__':
    main()