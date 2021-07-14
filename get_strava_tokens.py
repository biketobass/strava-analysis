import requests
import json


# This script simply gets access and refresh tokens and saves them to a file.
# You will only need to run it once after going through the strava authorization
# process. That process gives you the necessary authorization code.
# Check the Readme and related blog post for details. Replace your client_id,
# client_secret, and code with your ID, secret, and authorization code respectively.
# Note that client_id is an integer. The other two are strings. This script also saves
# the ID, secret, and code in a file called secret_stuff.json so that the main file
# strava.py can access them.

secret_stuff = {'client_id': [REPLACE WITH YOUR CLIENT ID], 'client_secret': '[REPLACE WITH YOUR CLIENT SECRET]', 'code': '[REPLACE WITH YOUR AUTHORIZATION CODE]', 'grant_type': 'authorization_code'}

# Get some tokens.
response = requests.post(url = 'https://www.strava.com/oauth/token',
                         data = secret_stuff)

# Get the tokens from the response's json
strava_tokens = response.json()
# Save them to a file
with open('strava_tokens.json', 'w') as outfile:
    json.dump(strava_tokens, outfile)

# While we're at it, save the secret_stuff to a file as well
# to be used later in the StravaAnalyzer class.
with open('secret_stuff.json', 'w') as outfile:
    json.dump(secret_stuff, outfile)
