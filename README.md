# strava-analysis

This repository contains code for downloading and analyzing one's data on Strava. It currently is able to
* download all of one's activities on Strava and save them to a CSV file
* update that CSV with new activities without downloading anything that is already stored locally
* create a CSV file with summary data for each activity type.
* separate the activities by type (Ride, Hike, etc.) and create a CSV file for each type
* produce a CSV file for each activity type with summary stats by year
* use past performance to predict the average speed of a route based on its distance and elevation gain

If you want to use it make sure that you have the following packages installed:
* Pandas - the code relies on pandas DataFrames heavily for manipulating the data
* requests - for making calls to the Strava API
* dateutil - for handling human readable dates
* pytz - for timezone support
* matplotlib
* numpy
* scipy
* scikit-learn

Also make sure that you have `__init__.py` in the same directory as the over python files. This is an empty file whose sole purpose is to let python know to look in the directory when `strava` is imported.

Getting the code to work is a two step process. See my [blog post about it](https://biketobass.github.io/computer/python/software/data/strava/fitness/2021/07/13/using-python-to-analyze-strava-data.html) for details. Step 1 which you only have to do once is to go through Strava's OAuth 2.0 authorization process. Towards the end of this step you will use `get_strava_tokens.py` to retrieve and store the access and refresh tokens that the rest of the code requires. Step 2 is simply to run `get_strava_activities.py` which instantiates a StravaAnalyzer object as defined in `strava.py`. The first time a StravaAnalyzer is created, the constructor accesses the necessary tokens in order to make a Strava API request. It then downloads all of your activities from the beginning of time to the present and saves them in a CSV file called `strava-activties.csv`. After that, it generates a CSV for each activity type you've recorded (example: Ride.csv, Hike.csv, Kayaking.csv, etc.). It also generates a high level summary CSV file called `strava-summary.csv` and creates a file with annual statistics for each type (Ride-by-year.csv, Hike-by-year.csv, etc.).

As long as you don't delete `strava-activities.csv`, the next time you create an instance of `StravaAnalyzer`, it will check for new activities, append them to `strava-activities.csv` and regenerate the other CSV files. If you do delete `strava-activities.csv`, it downloads everything again for you.


If you'd like see to what profile information Strava stores for you, uncomment the next line of code in `get_strava_activities.py` which calls `get_strava_profile()` which produces a file called `profile.csv` that contains all of the your profile information.

Finally, if you are a cyclist and want a prediction of your average speed given the distance and total elevation gain of a route, run the method `predict_avg_speed()`. Uncommenting the first call of this method in `get_strava_activities.py` runs an example in Imperial units (feet of elevation gain and miles of distance). The second call is the same ride but in metric units (meters of elevation gain and kilometers of distance).  See my [blog post](https://biketobass.github.io/computer/python/software/data/strava/fitness/2021/07/13/using-python-to-analyze-strava-data.html) for more details about this method.
