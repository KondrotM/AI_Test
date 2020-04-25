import requests
from contextlib import closing
import csv

url = "http://environment.data.gov.uk/flood-monitoring/archive/readings-2020-04-18.csv"

with closing(requests.get(url, stream=True)) as r:
    reader = csv.reader(r.iter_lines(), delimiter=',', quotechar='"')
    for row in reader:
        # Handle each row here...
        print( row  ) 