import requests

url = "https://lknpd.nalog.ru/api/v1/receipt/022200790955/2000xlf2lf/print"
payload = {'url': url}
r = requests.get('http://127.0.0.1:5000/get_check_data', params=payload)
print(r.text)


