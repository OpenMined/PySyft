import pandas as pd
import os

import syft as sy
from syft import autocache

DATA_SCIENTIST_EMAIL = "jane@caltech.edu"
DATA_SCIENTIST_PASSWORD = "abc123"

# notebooks/api/0.8/00-load-data.ipynb
# Uploading a private dataset as a Data Owner
def data_owner_uploads_data_to_node(data_owner):
    country = sy.DataSubject(name="Country", aliases=["country_code"])
    canada = sy.DataSubject(name="Canada", aliases=["country_code:ca"])
    country.add_member(canada)
    response = data_owner.data_subject_registry.add_data_subject(country)
    print(response)
    data_subjects = data_owner.data_subject_registry.get_all()
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

    upload_res = data_owner.upload_dataset(dataset)
    print(upload_res)
    datasets = data_owner.datasets.get_all()
    print(datasets)

    mock = data_owner.datasets[0].assets[0].mock
    print(mock)
    real = data_owner.datasets[0].assets[0].data
    print(real)

    # register a data scientist
    # domain_client.settings.allow_guest_signup(enable=True)
    data_owner.register(
        name="Jane Doe",
        email=DATA_SCIENTIST_EMAIL,
        password=DATA_SCIENTIST_PASSWORD,
        password_verify=DATA_SCIENTIST_PASSWORD,
        institution="Caltech",
        website="https://www.caltech.edu/",
    )

# notebooks/api/0.8/01-submit-code.ipynb
# Submitting code to run analysis on the private dataset as a Data Scientist
def data_scientist_requests_code_execution(domain_client):
    data_scientist = domain_client.login(
        email=DATA_SCIENTIST_EMAIL,
        password=DATA_SCIENTIST_PASSWORD,
    )

    data_scientist.me.set_payment_auth_token(os.environ.get('DATA_SCIENTIST_AUTH_TOKEN'))

    results = data_scientist.datasets.get_all()
    dataset = results[0]
    asset = dataset.assets[0]
    mock = asset.mock
    print(asset.data) # cannot access the private data

    print(mock["Trade Value (US$)"].sum())

    # We wrap our compute function with this decorator to make the function run exactly on the `asset` dataset
    # This converts the function into a SubmitUserCode object
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
    # TODO: confirm this interpretation of the following code:
    # Validate code against the mock data, on a mock server,
    # before submitting it to the Domain Server
    #####################

    pointer = sum_trade_value_mil(trade_data=asset)
    result = pointer.get()
    print(result[0])

    print(sum_trade_value_mil.code)

    #####################
    # Submit code to the Domain Server
    #####################

    new_project = sy.Project(
        name="My Cool UN Project",
        description="Hi, I want to calculate the trade volume in million's with my cool code.",
        members=[data_scientist],
    )
    print(new_project)

    # on the node, parse the code, convert to byte-code, send a notification,
    # on the client, create the RemoteUserCodeFunction: jane_client.code.sum_trade_value_mil
    result = new_project.create_code_request(sum_trade_value_mil, data_scientist)
    print(result)

    # Not clear what this line does:
    project = new_project.send()
    print(project)

# notebooks/api/0.8/02-review-code-and-approve.ipynb
# review and run code as data owner
def data_owner_reviews_and_runs_code(data_owner):
    # review data scientist's code
    # While Syft makes sure that the function is not tampered with,
    # it does not perform any validation on the implementation itself.
    # It is the Data Owner's responsibility to review the code & verify if it's safe to execute.
    project = data_owner.projects[0]
    request = project.requests[0]
    func = request.code
    print(func)

    # review data that the code will run on
    asset = func.assets[0]
    pvt_data = asset.data
    print(pvt_data)

    # execute the data scientist's code
    users_function = func.unsafe_function
    real_result = users_function(trade_data=pvt_data)

    # share result with data scientist
    # request object also has “approve” (which is called by “accept_by_depositing_result") and “approve_with_client”
    result = request.accept_by_depositing_result(real_result, force=True)
    print(result)

# notebooks/api/0.8/03-data-scientist-download-result.ipynb
# Data Scientist downloads the result
def data_scientist_downloads_result(domain_client):
    data_scientist = domain_client.login(
        email=DATA_SCIENTIST_EMAIL,
        password=DATA_SCIENTIST_PASSWORD,
    )

    asset = data_scientist.datasets[0].assets[0]

    # this attempt to execute the code on the server will succeed
    # ... as the code is approved by the data owner:
    result_pointer = data_scientist.code.sum_trade_value_mil(trade_data=asset)
    real_result = result_pointer.get()
    print(real_result)

def main():
    sy.requires(">=0.8.6,<0.8.7")

    # Log into the node with default root credentials
    domain_client = sy.login(
        port=8080,
        email="info@openmined.org",
        password="changethis",
    )

    domain_client.me.set_payment_auth_token(os.environ.get('DATA_OWNER_AUTH_TOKEN'))

    data_owner_uploads_data_to_node(domain_client)
    data_scientist_requests_code_execution(domain_client)
    data_owner_reviews_and_runs_code(domain_client)
    data_scientist_downloads_result(domain_client)

if __name__ == '__main__':
    main()
