import tweepy
import pandas as pd
import os

def coletar_tweets(bearer_token: str, hashtag: str, idioma: str = "pt", limite: int = 1000):
    """
    Coleta tweets com base em uma hashtag e idioma.

    Args:
        bearer_token (str): Token da API do Twitter.
        hashtag (str): Termo ou hashtag para busca.
        idioma (str): Código do idioma (ex: 'pt', 'en').
        limite (int): Número máximo de tweets.
    """
    client = tweepy.Client(bearer_token=bearer_token)

    query = f"{hashtag} -is:retweet lang:{idioma}"
    tweets = []

    for tweet in tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["text"],
        max_results=100
    ).flatten(limit=limite):
        tweets.append(tweet.text)

    df = pd.DataFrame(tweets, columns=["text"])
    os.makedirs("dados", exist_ok=True)
    df.to_csv("dados/tweets.csv", index=False)
    print(f"[✓] {len(df)} tweets salvos em dados/tweets.csv")

