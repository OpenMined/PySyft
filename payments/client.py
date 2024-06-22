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
  jane_client = domain_client.login(email="jane@caltech.edu", password="abc123")

  results = jane_client.datasets.get_all()
  dataset = results[0]
  asset = dataset.assets[0]
  mock = asset.mock
  print(asset.data) # cannot access the private data

  print(mock["Trade Value (US$)"].sum())

  # We wrap our compute function with this decorator to make the function run exactly on the `asset` dataset
  @sy.syft_function_single_use(trade_data=asset)
  def sum_trade_value_mil(trade_data):
      # third party
      import opendp.prelude as dp

      dp.enable_features("contrib")

      aggregate = 0.0
      base_lap = dp.m.make_base_laplace(
          dp.atom_domain(T=float),
          dp.absolute_distance(T=float),
          scale=5.0,
      )
      noise = base_lap(aggregate)

      df = trade_data
      total = df["Trade Value (US$)"].sum()
      return (float(total / 1_000_000), float(noise))

  #####################
  # TODO: confirm: 
  # Validate code against the mock data, on a mock server, 
  # before submitting it to the Domain Server
  #####################

  # pointer = sum_trade_value_mil(trade_data=asset)
  # result = pointer.get()
  # print(result[0])

  # print(sum_trade_value_mil.code) 

  #####################
  # Submit code to the Domain Server
  #####################

  new_project = sy.Project(
      name="My Cool UN Project",
      description="Hi, I want to calculate the trade volume in million's with my cool code.",
      members=[jane_client],
  )
  print(new_project)

  # this parses the code on the node, converts to byte-code, and sends a notification: 
  result = new_project.create_code_request(sum_trade_value_mil, jane_client)
  print(result)

  # Not clear what this line does: 
  project = new_project.send()
  print(project)

  #####################
  # Running the Syft Function on the Domain Server
  #####################

  # this attempt to execute the code on the server will fail 
  # ... as the code is not approved yet by the data owner: 
  result = jane_client.code.sum_trade_value_mil(trade_data=asset) 
  print(result)

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
