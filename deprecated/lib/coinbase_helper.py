from coinbase.wallet.client import Client


class CoinbaseHelper():
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_price(self, seconds):

        price = self.client.get_spot_price(currency_pair='ETH-USD')

        dollar_per_second = 0.00025
        ether_per_dollar = 1 / float(price['amount'])
        ether_price_per_second = ether_per_dollar * dollar_per_second
        final_price = ether_price_per_second * seconds

        if final_price < 0.000001:
            print("SENDING 0.000001")
            return 0.000001

        print(f"SENDING {final_price}")
        return final_price

    def send_ether(self, email, seconds):
        account = self.client.get_account("ETH")
        amount_to_send = self.get_price(seconds)
        tx = account.send_money(
            to=email, amount=amount_to_send, currency='ETH')

        print(tx)
