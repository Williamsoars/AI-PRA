import tweepy
import pandas as pd
from pathlib import Path
import logging

def collect_tweets(
    bearer_token: str,
    hashtag: str,
    language: str = "pt",
    limit: int = 1000,
    output_path: Path = None
):
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        query = f"{hashtag} -is:retweet lang:{language}"
        tweets = []

        for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            tweet_fields=["text"],
            max_results=100
        ).flatten(limit=limit):
            tweets.append(tweet.text)

        df = pd.DataFrame(tweets, columns=["text"])
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"{len(df)} tweets salvos em {output_path}")
        return df
        
    except Exception as e:
        logging.error(f"Erro na coleta de tweets: {str(e)}")
        raise

