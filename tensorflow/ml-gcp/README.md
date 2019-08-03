# Intro to Tensorflow on GCP

```shell
rm -rf earthquake.csv
# download latest data from USGS
wget http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv -O earthquakes.csv
gsutil cp earthquakes.* gs://<YOUR-BUCKET>/earthquakes/
# publish cloud files to the web
gsutil acl ch -u AllUsers:R gs://<YOUR-BUCKET>/earthquakes/*
# get compute zones names
gcloud compute zones list
datalab create mydatalabvm --zone <ZONE> # create datalab instance in the zone of your data
# This takes some time, once ready you'll see Web preview enabled in Cloud shell
# If cloud shell is closed for some reason as they are ephermeral. We can reconnect to the same datalab instance using
datalab connect mydatalabvm
# Enable Cloud Source API
```

[Reading CSV files](earthquakes.ipynb)

Check [flights notebook](flights.ipynb) to see invocation of BigQuery and getting results in Pandas dataframes.

