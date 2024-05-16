import requests

proxies = [dict(http='http://10.0.3.1:1080', https='http://10.0.3.1:1080'),
           dict(http='http://10.0.3.1:1081', https='http://10.0.3.1:1081'),
           dict(http='http://10.0.3.1:1082', https='http://10.0.3.1:1082'),
           dict(http='http://10.0.3.1:1083', https='http://10.0.3.1:1083'),
           dict(http='http://10.0.3.1:1084', https='http://10.0.3.1:1084'),
           dict(http='http://10.0.3.1:1085', https='http://10.0.3.1:1085'),
           dict(http='http://10.0.3.1:1086', https='http://10.0.3.1:1086'),
           dict(http='http://10.0.3.1:1087', https='http://10.0.3.1:1087')]

for proxy in proxies:

    print(requests.get("https://api.myip.com",proxies=proxy).content)