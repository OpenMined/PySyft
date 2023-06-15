```python
SYFT_VERSION = ">=0.8.1b0,<0.9"
package_string = f'"syft{SYFT_VERSION}"'
%pip install {package_string} -f https://whls.blob.core.windows.net/unstable/index.html -q
```

    Note: you may need to restart the kernel to use updated packages.



```python
import syft as sy
sy.requires(SYFT_VERSION)
from syft import autocache
```

    kj/filesystem-disk-unix.c++:1703: warning: PWD environment variable doesn't match current directory; pwd = /home/shubham/PySyft


    ✅ The installed version of syft==0.8.1b9 matches the requirement >=0.8.1b0 and the requirement <0.9



```python
node = sy.orchestra.launch(name="test-domain-1", port="auto", dev_mode=True, reset=True)
```

    Starting test-domain-1 server on 0.0.0.0:6032
    
    WARNING: private key is based on node name: test-domain-1 in dev_mode. Don't run this in production.
    SQLite Store Path:
    !open file:///tmp/7bca415d13ed4ec881f0d0aede098dbb.sqlite
    


    INFO:     Started server process [22474]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:6032 (Press CTRL+C to quit)


    INFO:     127.0.0.1:36606 - "GET /api/v2/metadata HTTP/1.1" 200 OK
    Waiting for server to start Done.
    INFO:     127.0.0.1:36608 - "GET /api/v2/metadata HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36608 - "POST /api/v2/login HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36608 - "GET /api/v2/api?verify_key=aec6ea4dfc049ceacaeeebc493167a88a200ddc367b1fa32da652444b635d21f HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36624 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36626 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36630 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36638 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41944 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41952 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41960 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41976 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41988 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:41992 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:42000 - "POST /api/v2/api_call HTTP/1.1" 200 OK
    INFO:     127.0.0.1:36608 - "POST /api/v2/register HTTP/1.1" 200 OK


    INFO:     Shutting down
    INFO:     Waiting for application shutdown.
    INFO:     Application shutdown complete.
    INFO:     Finished server process [22474]



```python
domain_client = node.login(email="info@openmined.org", password="changethis")
```

    Logged into test-domain-1 as GUEST



```python
domain_client.api
```




```python
class SyftAPI:
  id: str = 91564498511943c5b0f7023e2e7d5d38

```




```python
data_subjects = domain_client.api.services.data_subject.get_all()
```


```python
data_subjects
```




[]




```python
assert len(data_subjects) == 0
```


```python
country = sy.DataSubject(name="Country", aliases=["country_code"])
```


```python
canada = sy.DataSubject(name="Canada", aliases=["country_code:ca"])
germany = sy.DataSubject(name="Germany", aliases=["country_code:de"])
spain = sy.DataSubject(name="Spain", aliases=["country_code:es"])
france = sy.DataSubject(name="France", aliases=["country_code:fr"])
japan = sy.DataSubject(name="Japan", aliases=["country_code:jp"])
uk = sy.DataSubject(name="United Kingdom", aliases=["country_code:uk"])
usa = sy.DataSubject(name="United States of America", aliases=["country_code:us"])
australia = sy.DataSubject(name="Australia", aliases=["country_code:au"])
india = sy.DataSubject(name="India", aliases=["country_code:in"])
```


```python
country.add_member(canada)
country.add_member(germany)
country.add_member(spain)
country.add_member(france)
country.add_member(japan)
country.add_member(uk)
country.add_member(usa)
country.add_member(australia)
country.add_member(india)

country.members
```





                <style>
                .syft-collection-header {color: #464158;}
                </style>
                <div class='syft-collection-header'>
                    <h3>DataSubjectCreate Dict</h3>
                </div>
                <br>
                <style>
.itables table {
    margin: 0 auto;
    float: left;
    color: #534F64;
}
.itables table th {color: #2E2B3B;}
</style>
<div class="itables">
<table id="5405d692-8a6b-41e2-8c2d-b7e72933eba0" class="display nowrap"style="table-layout:auto;width:auto;margin:auto;caption-side:bottom"><thead>
    <tr style="text-align: right;">

      <th>key</th>
    </tr>
  </thead><tbody><tr><td>Loading... (need <a href=https://mwouts.github.io/itables/troubleshooting.html>help</a>?)</td></tr></tbody></table>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css">
<script type="module">
    // Import jquery and DataTable
    import 'https://code.jquery.com/jquery-3.6.0.min.js';
    import dt from 'https://cdn.datatables.net/1.12.1/js/jquery.dataTables.mjs';
    dt($);

    // Define the table data
    const data = [["Canada"], ["Germany"], ["Spain"], ["France"], ["Japan"], ["United Kingdom"], ["United States of America"], ["Australia"], ["India"]];

    // Define the dt_args
    let dt_args = {"order": [], "dom": "t"};
    dt_args["data"] = data;

    $(document).ready(function () {

        $('#5405d692-8a6b-41e2-8c2d-b7e72933eba0').DataTable(dt_args);
    });
</script>
</div>





```python
registry = domain_client.data_subject_registry
```


```python
response = registry.add_data_subject(country)
```


```python
response
```




<div class="alert-success" style="padding:5px;"><strong>SyftSuccess</strong>: 10 Data Subjects Registered</div><br />




```python
assert response
```


```python
domain_client.data_subject_registry
```





                <style>
                .syft-collection-header {color: #464158;}
                </style>
                <div class='syft-collection-header'>
                    <h3>DataSubject List</h3>
                </div>
                <br>
                <style>
.itables table {
    margin: 0 auto;
    float: left;
    color: #534F64;
}
.itables table th {color: #2E2B3B;}
</style>
<div class="itables">
<table id="165864dc-d90e-4164-9bd8-74ef6d0a841b" class="display nowrap"style="table-layout:auto;width:auto;margin:auto;caption-side:bottom"><thead>
    <tr style="text-align: right;">

      <th>id</th>
    </tr>
  </thead><tbody><tr><td>Loading... (need <a href=https://mwouts.github.io/itables/troubleshooting.html>help</a>?)</td></tr></tbody></table>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css">
<script type="module">
    // Import jquery and DataTable
    import 'https://code.jquery.com/jquery-3.6.0.min.js';
    import dt from 'https://cdn.datatables.net/1.12.1/js/jquery.dataTables.mjs';
    dt($);

    // Define the table data
    const data = [["5712...e9a"], ["fe56...7d3"], ["1fff...b03"], ["ba2f...277"], ["4f77...3cc"], ["43ce...fb1"], ["b9e0...3d8"], ["7c3f...f72"], ["d2e9...3fe"], ["d311...78d"]];

    // Define the dt_args
    let dt_args = {"order": [], "dom": "t"};
    dt_args["data"] = data;

    $(document).ready(function () {

        $('#165864dc-d90e-4164-9bd8-74ef6d0a841b').DataTable(dt_args);
    });
</script>
</div>





```python
data_subjects = domain_client.api.services.data_subject.get_all()
```


```python
assert len(data_subjects) == 10
```


```python
canada_dataset_url = "https://github.com/OpenMined/datasets/blob/main/trade_flow/ca%20-%20feb%202021.csv?raw=True"
```


```python
import pandas as pd
```


```python
df = pd.read_csv(autocache(canada_dataset_url))
```

    /tmp/ipykernel_22427/754433127.py:1: DtypeWarning: Columns (14) have mixed types. Specify dtype option on import or set low_memory=False.
      df = pd.read_csv(autocache(canada_dataset_url))



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classification</th>
      <th>Year</th>
      <th>Period</th>
      <th>Period Desc.</th>
      <th>Aggregate Level</th>
      <th>Is Leaf Code</th>
      <th>Trade Flow Code</th>
      <th>Trade Flow</th>
      <th>Reporter Code</th>
      <th>Reporter</th>
      <th>...</th>
      <th>Partner</th>
      <th>Partner ISO</th>
      <th>Commodity Code</th>
      <th>Commodity</th>
      <th>Qty Unit Code</th>
      <th>Qty Unit</th>
      <th>Qty</th>
      <th>Netweight (kg)</th>
      <th>Trade Value (US$)</th>
      <th>Flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Other Asia, nes</td>
      <td>NaN</td>
      <td>6117</td>
      <td>Clothing accessories; made up, knitted or croc...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9285</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Egypt</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>116604</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1495175</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>United Rep. of Tanzania</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2248</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Singapore</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>47840</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>227449</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>Exports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>World</td>
      <td>NaN</td>
      <td>550952</td>
      <td>Yarn; (not sewing thread), of polyester staple...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5406.0</td>
      <td>34272</td>
      <td>0</td>
    </tr>
    <tr>
      <th>227450</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>Exports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>World</td>
      <td>NaN</td>
      <td>550999</td>
      <td>Yarn; (not sewing thread), of synthetic staple...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7345.0</td>
      <td>228182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>227451</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>Exports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>World</td>
      <td>NaN</td>
      <td>550969</td>
      <td>Yarn; (not sewing thread), of acrylic or modac...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1839.0</td>
      <td>18812</td>
      <td>0</td>
    </tr>
    <tr>
      <th>227452</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>Exports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>World</td>
      <td>NaN</td>
      <td>550962</td>
      <td>Yarn; (not sewing thread), of acrylic or modac...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1437.0</td>
      <td>23140</td>
      <td>0</td>
    </tr>
    <tr>
      <th>227453</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>Exports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>World</td>
      <td>NaN</td>
      <td>550959</td>
      <td>Yarn; (not sewing thread), of polyester staple...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5660.0</td>
      <td>73652</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>227454 rows × 22 columns</p>
</div>




```python
ca_data = df[0:10]
```


```python
mock_ca_data = df[10:20]
```


```python
ca_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classification</th>
      <th>Year</th>
      <th>Period</th>
      <th>Period Desc.</th>
      <th>Aggregate Level</th>
      <th>Is Leaf Code</th>
      <th>Trade Flow Code</th>
      <th>Trade Flow</th>
      <th>Reporter Code</th>
      <th>Reporter</th>
      <th>...</th>
      <th>Partner</th>
      <th>Partner ISO</th>
      <th>Commodity Code</th>
      <th>Commodity</th>
      <th>Qty Unit Code</th>
      <th>Qty Unit</th>
      <th>Qty</th>
      <th>Netweight (kg)</th>
      <th>Trade Value (US$)</th>
      <th>Flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Other Asia, nes</td>
      <td>NaN</td>
      <td>6117</td>
      <td>Clothing accessories; made up, knitted or croc...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9285</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Egypt</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>116604</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1495175</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>United Rep. of Tanzania</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2248</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Singapore</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>47840</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Viet Nam</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3526</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>South Africa</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5462</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Spain</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>311425</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Sweden</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>11786</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HS</td>
      <td>2021</td>
      <td>202102</td>
      <td>February 2021</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Imports</td>
      <td>124</td>
      <td>Canada</td>
      <td>...</td>
      <td>Venezuela</td>
      <td>NaN</td>
      <td>18</td>
      <td>Cocoa and cocoa preparations</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>33715</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 22 columns</p>
</div>




```python
dataset = sy.Dataset(name="Canada Trade Value")
```


```python
dataset.set_description("""Canada Trade Data""")
```


```python
dataset.add_citation("Person, place or thing")
dataset.add_url("https://github.com/OpenMined/datasets/tree/main/trade_flow")
```


```python
dataset.add_contributor(role=sy.roles.UPLOADER, 
                                name="Andrew Trask", 
                                email="andrew@openmined.org",
                                note="Andrew runs this domain and prepared the dataset metadata.")

dataset.add_contributor(role=sy.roles.EDITOR, 
                                name="Madhava Jay", 
                                email="madhava@openmined.org",
                                note="Madhava tweaked the description to add the URL because Andrew forgot.")
```


```python
dataset.contributors
```





                <style>
                .syft-collection-header {color: #464158;}
                </style>
                <div class='syft-collection-header'>
                    <h3>Contributor List</h3>
                </div>
                <br>
                <style>
.itables table {
    margin: 0 auto;
    float: left;
    color: #534F64;
}
.itables table th {color: #2E2B3B;}
</style>
<div class="itables">
<table id="2b19cf82-e18e-4bd8-8872-033f4f47ff4c" class="display nowrap"style="table-layout:auto;width:auto;margin:auto;caption-side:bottom"><thead>
    <tr style="text-align: right;">

      <th>id</th>
      <th>name</th>
      <th>role</th>
      <th>email</th>
    </tr>
  </thead><tbody><tr><td>Loading... (need <a href=https://mwouts.github.io/itables/troubleshooting.html>help</a>?)</td></tr></tbody></table>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css">
<script type="module">
    // Import jquery and DataTable
    import 'https://code.jquery.com/jquery-3.6.0.min.js';
    import dt from 'https://cdn.datatables.net/1.12.1/js/jquery.dataTables.mjs';
    dt($);

    // Define the table data
    const data = [["6931...e98", "Andrew Trask", "Uploader", "andrew@openmined.org"], ["c9c7...202", "Madhava Jay", "Editor", "madhava@openmined.org"]];

    // Define the dt_args
    let dt_args = {"order": [], "dom": "t"};
    dt_args["data"] = data;

    $(document).ready(function () {

        $('#2b19cf82-e18e-4bd8-8872-033f4f47ff4c').DataTable(dt_args);
    });
</script>
</div>





```python
assert len(dataset.contributors) == 2
```


```python
ctf = sy.Asset(name="canada_trade_flow")
ctf.set_description("""all the datas""")
```


```python
ctf.add_contributor(role=sy.roles.UPLOADER, 
                      name="Andrew Trask", 
                      email="andrew@openmined.org",
                      note="Andrew runs this domain and prepared the asset.")
```


```python
ctf.set_obj(ca_data)
```


```python
ctf.set_shape((10, 22))
```


```python
ctf.add_data_subject(canada)
```


```python
ctf.no_mock()
```


```python
dataset.add_asset(ctf)
```


```python
dataset.remove_asset(name=ctf.name)
```


```python
ctf.set_mock(mock_ca_data, mock_is_real=False)
```


```python
dataset.add_asset(ctf)
```


```python
domain_client.upload_dataset(dataset)
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 10.61it/s]

    Uploading: canada_trade_flow


    





<div class="alert-success" style="padding:5px;"><strong>SyftSuccess</strong>: Dataset Added</div><br />




```python
datasets = domain_client.api.services.dataset.get_all()
```


```python
assert len(datasets) == 1
```


```python
datasets
```





                <style>
                .syft-collection-header {color: #464158;}
                </style>
                <div class='syft-collection-header'>
                    <h3>Dataset List</h3>
                </div>
                <br>
                <style>
.itables table {
    margin: 0 auto;
    float: left;
    color: #534F64;
}
.itables table th {color: #2E2B3B;}
</style>
<div class="itables">
<table id="e95a24f4-ff38-4ae2-b240-158e32757307" class="display nowrap"style="table-layout:auto;width:auto;margin:auto;caption-side:bottom"><thead>
    <tr style="text-align: right;">

      <th>id</th>
      <th>name</th>
      <th>url</th>
    </tr>
  </thead><tbody><tr><td>Loading... (need <a href=https://mwouts.github.io/itables/troubleshooting.html>help</a>?)</td></tr></tbody></table>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css">
<script type="module">
    // Import jquery and DataTable
    import 'https://code.jquery.com/jquery-3.6.0.min.js';
    import dt from 'https://cdn.datatables.net/1.12.1/js/jquery.dataTables.mjs';
    dt($);

    // Define the table data
    const data = [["0851...22c", "Canada Trade Value", "https://github.com/OpenMined/datasets/tree/main/trade_flow"]];

    // Define the dt_args
    let dt_args = {"order": [], "dom": "t"};
    dt_args["data"] = data;

    $(document).ready(function () {

        $('#e95a24f4-ff38-4ae2-b240-158e32757307').DataTable(dt_args);
    });
</script>
</div>





```python
mock = domain_client.datasets[0].assets[0].mock
```


```python
assert mock_ca_data.equals(mock)
```


```python
real = domain_client.datasets[0].assets[0].data
```


```python
assert ca_data.equals(real.syft_action_data)
```


```python
### Create account for guest user
### Signup is disabled by default
### An Admin/DO can enable it by `domain_client.settings.allow_guest_signup(enable=True)`

domain_client.register(name="Jane Doe", email="jane@caltech.edu", password="abc123", institution="Caltech", website="https://www.caltech.edu/")
```




<div class="alert-success" style="padding:5px;"><strong>SyftSuccess</strong>: User successfully registered!</div><br />




```python
if node.node_type.value == "python":
    node.land()
```

    Stopping test-domain-1



```python

```
