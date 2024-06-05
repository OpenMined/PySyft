import pandas as pd

import syft as sy
from syft import autocache

# notebooks/api/0.8/00-load-data.ipynb  
# Uploading a private dataset as a Data Owner
def upload_data_to_node(domain_client):   
  country = sy.DataSubject(name="Country", aliases=["country_code"])
  canada = sy.DataSubject(name="Canada", aliases=["country_code:ca"])
  country.add_member(canada)
  response = domain_client.data_subject_registry.add_data_subject(country)
  print(response)
  data_subjects = domain_client.data_subject_registry.get_all()
  print(data_subjects)

  dataset = sy.Dataset(name="Canada Trade Value")
  dataset.set_description("Canada Trade Data")

  canada_dataset_url = "https://github.com/OpenMined/datasets/blob/main/trade_flow/ca%20-%20feb%202021.csv?raw=True"
  df = pd.read_csv(autocache(canada_dataset_url))
  ca_data = df[0:10]
  mock_ca_data = df[10:20]

  ctf = sy.Asset(name="canada_trade_flow")
  ctf.set_description(
      "Canada trade flow represents export & import of different commodities to other countries"
  )
  ctf.set_obj(ca_data)
  ctf.set_shape(ca_data.shape)
  ctf.add_data_subject(canada)
  ctf.set_mock(mock_ca_data, mock_is_real=False)
  
  dataset.add_asset(ctf)

  upload_res = domain_client.upload_dataset(dataset)
  print(upload_res)
  datasets = domain_client.datasets.get_all()
  print(datasets)

  mock = domain_client.datasets[0].assets[0].mock
  print(mock)
  real = domain_client.datasets[0].assets[0].data
  print(real)

  # domain_client.settings.allow_guest_signup(enable=True)
  domain_client.register(
    name="Jane Doe",
    email="jane@caltech.edu",
    password="abc123",
    password_verify="abc123",
    institution="Caltech",
    website="https://www.caltech.edu/",
  )

# notebooks/api/0.8/01-submit-code.ipynb
# Submitting code to run analysis on the private dataset as a Data Scientist
def submit_code(domain_client):
  pass 
      
def main(): 
  sy.requires(">=0.8.6,<0.8.7")
  
  domain_client = sy.login(
    port=8080,
    email="info@openmined.org",
    password="changethis"
  )

  upload_data_to_node(domain_client)
  
  submit_code(domain_client)

if __name__ == '__main__':
  main()
