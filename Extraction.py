import tweepy
import pandas as pd

# Substitua com seu token
bearer_token = "SEU_BEARER_TOKEN"

client = tweepy.Client(bearer_token=bearer_token)

query = "#examplehashtag -is:retweet lang:pt"
tweets = []

# Coleta de até 10.000 tweets (100 por página)
for tweet in tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["text"],
        max_results=100
    ).flatten(limit=10000):
    tweets.append(tweet.text)

# Salvando em CSV
df = pd.DataFrame(tweets, columns=["text"])
df.to_csv("tweets.csv", index=False)
print("Tweets salvos com sucesso.")
