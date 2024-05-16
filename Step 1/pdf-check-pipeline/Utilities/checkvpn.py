import random
import requests
from collections import defaultdict
from ipaddress import ip_address

class CheckVPN:

    def __init__(self):
        self.ip_services = [
            'http://icanhazip.com',
            'https://ipecho.net/plain',
            'http://ifconfig.me',
            'http://checkip.amazonaws.com',
            'http://whatismyip.akamai.com',
            'http://api.infoip.io/ip',
            'https://api.ipify.org/',
        ]
        self.endpoint = random.choice(self.ip_services)
        self.retries = defaultdict(int)


    def check_one_vpn(self, i, proxy=None):
        print(self.endpoint)

        try:
            if proxy:
                response = requests.get(self.endpoint, proxies=proxy)
            else:
                response = requests.get(self.endpoint)

            vpn_ip = ip_address(response.content.decode('utf-8').strip())

            print(f"VPN:{i}, LocalIp: {proxy}, RetrievedIp:{vpn_ip}")

        except ValueError as e:
            print(e)
            print(f'Website did not return an ip:\n\t{response.content[:500]}')

            self.retries[self.endpoint] += 1

            self.endpoint = random.choice(self.ip_services)
            while self.retries[self.endpoint] > 2:
                ip_services.remove(self.endpoint)
                try:
                    self.endpoint = random.choice(self.endpoint)
                except IndexError as e:
                    print(e)
                    sys.exit (-1)

        except requests.RequestException as e1:
            print(e1)
            print(f'Error with service {self.endpoint} or proxy {proxy}')

            self.retries[self.endpoint] += 1

            self.endpoint = random.choice(self.ip_services)
            while self.retries[self.endpoint] > 2:
                ip_services.remove(self.endpoint)
                try:
                    self.endpoint = random.choice(self.endpoint)
                except IndexError as e:
                    print(e)
                    raise Exception('No more endpoints to try!')


