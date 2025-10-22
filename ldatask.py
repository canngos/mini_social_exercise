import re

DATABASE = 'database.sqlite'

def sentiment_algorithm(contents):
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()

    # sentiment scores for posts and comments
    score = [sia.polarity_scores(content)['compound'] for content in contents]

    # Overall Platform Tone
    if len(score) > 0:
        overall_sentiment = sum(score) / len(score)
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


def main():
    import sqlite3
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.models import LdaModel
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Connect to the database
    conn = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    # Fetch all post contents
    cursor.execute('SELECT content FROM posts')
    posts = cursor.fetchall()

    # Preprocess the text data
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    custom_stop_words = [
        'afternoon', 'also', 'always', 'amazing', 'another', 'anyone', 'article',
        'back', 'beat', 'believe', 'best', 'bought', 'change', 'could', 'damn',
        'day', 'else', 'even', 'every', 'fact', 'fascinating', 'feel', 'feeling',
        'finished', 'friend', 'fun', 'get', 'go', 'going', 'good', 'got', 'grateful',
        'great', 'host', 'incredible', 'keep', 'knew', 'know', 'last', 'let', 'life',
        'like', 'look', 'love', 'made', 'mail', 'make', 'many', 'mean', 'much',
        'need', 'never', 'new', 'night', 'nothing', 'old', 'one', 'people',
        'please', 'post', 'question', 'quick', 'reason', 'really', 'reflecting',
        'reply', 'said', 'see', 'small', 'sometimes', 'spent', 'thanks', 'thing',
        'think', 'thought', 'time', 'tried', 'true', 'trying', 'want',
        'well', 'work', 'world', 'would', 'year', 'today', 'find', 'wait', 'take',
        'latest', 'way', 'found', 'truly', 'perfect', 'real', 'still', 'fuck',
        'finally', 'everyone', 'weekend', 'shit', 'saw', 'took', 'give', 'first',
        'making', 'place', 'following', 'watched', 'attended', 'hard', 'little',
        'someone', 'helped', 'better', 'simple', 'next', 'university', 'together',
        'follow', 'enter', 'hit', 'air', 'moment', 'caught', 'local', 'doe', 'actually',
        'entered', 'com', 'stuff', 'favorite', 'something', 'cold', 'meme', 'difference',
        'hour', 'importance', 'beautiful', 'amaze', 'felt', 'tonight', 'stunning', 'gem',
        'hidden', 'cease', 'trip', 'happy', 'kid', 'story', 'step', 'joined', 'bring',
        'check', 'eye', 'morning', 'action', 'exploring', 'tip', 'discussion', 'perspective',
        'link', 'excited', 'lot', 'wakeup', 'hell', 'joy', 'shot', 'cup', 'view', 'celebrating',
        'heated', 'classic', 'coming', 'wonderful', 'delicious', 'fresh', 'taking', 'come', 'pushing',
        'spontaneous', 'session', 'initiative', 'energy', 'gon', 'solo', 'free', 'waiting', 'lie', 'miss',
        'journey', 'team', 'read', 'refreshing', '2025', 'needed', 'click', 'wonder', 'watching', 'mega',
        'freebie', 'everything', 'talk', 'mind', 'inspired', 'clean', 'fake', 'available', 'top', 'inspiring',
        'chat', 'loving', 'giving', 'piece', 'right', 'waste', 'nailed', 'surprise', 'store', 'ever', 'policy'
        'act', 'dive', 'gadget', 'moral'
    ]
    stop_words.extend(custom_stop_words)
    processed_posts = []
    for post in posts:
        text = post[0].lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 2]
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        if (len(filtered_tokens) > 0):
            processed_posts.append(filtered_tokens)

    # Create a dictionary and corpus for LDA
    dictionary = Dictionary(processed_posts)
    dictionary.filter_extremes(no_below=3, no_above=0.4)
    corpus = [dictionary.doc2bow(text) for text in processed_posts]

    # Build the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10, random_state=2)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_posts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence score:', coherence_lda)

    # Identify the 10 most popular topics discussed on our platform
    raw_topics = lda_model.print_topics(num_words=5)
    topic_details = {
        topic_num: ", ".join(re.findall(r'"(.*?)"', topic_words_str))
        for topic_num, topic_words_str in raw_topics
    }

    # Calculate the popularity of each topic
    topic_counts = [0] * 10
    for doc_bow in corpus:
        topic_distribution = lda_model.get_document_topics(doc_bow)
        dominant_topic = max(topic_distribution, key=lambda item: item[1])[0]
        topic_counts[dominant_topic] += 1

    # Create a sorted list of topics by popularity
    sorted_popularity = sorted(enumerate(topic_counts), key=lambda item: item[1], reverse=True)

    # ranked list of popular topics with their keywords
    print("\n----- 10 MOST POPULAR TOPICS -----")
    for topic_num, count in sorted_popularity:
        keywords = topic_details.get(topic_num, "not_found")
        print(f"Topic #{topic_num} ({count} posts): {keywords}")



    # Exercise 4.2 part
    # make sentiment analysis on each topic
    print("\n----- SENTIMENT ANALYSIS PER TOPIC -----")
    for topic_num, count in sorted_popularity:
        topic_posts = []
        for i, doc_bow in enumerate(corpus):
            topic_distribution = lda_model.get_document_topics(doc_bow)
            dominant_topic = max(topic_distribution, key=lambda item: item[1])[0]
            if dominant_topic == topic_num:
                original_post = posts[i][0]
                topic_posts.append(original_post)
        print(f"\nTopic #{topic_num} Sentiment Analysis:")
        sentiment_algorithm(topic_posts)

    # Close the database connection
    conn.close()


if __name__ == '__main__':
    main()