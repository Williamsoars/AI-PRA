from coleta.twitter_api import coletar_tweets

def main():
    print("=== Coletor de Tweets ===")
    hashtag = input("Digite a hashtag ou termo (ex: #bolsonaro): ")
    idioma = input("Digite o código do idioma (ex: pt, en, es): ").lower()
    limite = input("Quantos tweets você quer coletar? [padrão: 1000]: ")
    limite = int(limite) if limite.isdigit() else 1000

    # Substitua pelo seu Bearer Token real
    bearer_token = "SEU_TOKEN_AQUI"

    coletar_tweets(bearer_token, hashtag, idioma, limite)

if __name__ == "__main__":
    main()
