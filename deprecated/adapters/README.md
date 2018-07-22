# Grid Adapters

Proof of concepts for grid adapters.

### Twitter Example

For the twitter adapter, you must specify a config.json file in `grid/adapters/`.
See `grid/adapters/config.example.json` for an example of the required keys.

#### Generating keys

All twitter api requests must be authenticated.  To generate consumer and access token/secret
pairs.  Go to [apps.twitter.com](https://apps.twitter.com/) and click [create new app](https://apps.twitter.com/app/new).

Fill in the required info.  When you arrive on the app page after finishing, click the
`Keys and Access Tokens` tab.  There should already be an `Application Settings` section
that contains `consumer key` and `consumer secret` fields.

Copy `grid/adapters/config.example.json` to `grid/adapters/config.json` and fill in
`consumerKey` as well as `consumerSecret` (get these fields from the step above).

At the bottom of the page you will see a button to generate access token/secret pairs.
Click this and copy the access token and secret you get into `config.json`.  

Run `grid/adapters/tweets.py`.  If you see a big list of tweets, then you know that you
have set twitter up properly.
