import pandas as pd
import requests
import json
import time
import datetime
from dateutil import parser
import pytz
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn import linear_model

# This class defines various methods for accessing and analyzing data on Strava.

class StravaAnalyzer :

    # Constructor
    def __init__(self) :
        # When the object is created, the constructor makes sure it has the
        # the tokens necessary to connect to Strava and that they
        # are up to date. After updating as necessary, it stores the tokens
        # in a file called strava_tokens.json.
        # This file must already exist since the constructor first reads the
        # tokens from it and checks if they have expired. If the file does not yet exist,
        # the program displays a message to that effect and exits.
        # Run get_strava_tokens.py and try again. For more details see the Readme and
        # associated blog post.
        print("Checking if tokens have expired.")
        try :
            with open('strava_tokens.json') as json_file:
                self.strava_tokens = json.load(json_file)
        except FileNotFoundError:
            print('strava_tokens.json does not exist yet. Consult the Readme and run get_strava_tokens.py before using this class. You only need to run get_strava_tokens.py once. Exiting.')
            quit()

        # If access_token has expired, use the refresh_token to get the new access_token
        if self.strava_tokens['expires_at'] < time.time():
            print("They expired. Updating...")
            # Make Strava auth API call with current refresh token
            # Replace client_id and client_secret with yours. They are stored in
            # secret_stuff.json that was created when you ran get_strava_tokens.py.
            try:
                with open('secret_stuff.json') as secret_file:
                    secret_dict = json.load(secret_file)
            except FileNotFoundError:
                print("The client ID and client secret haven't been saved to disk yet. Consult the Readme and run get_strava_tokens.py before using this class. You only need to run get_strava_tokens.py once. Exiting.")
                quit()

            # Make the request and get the response..
            response = requests.post(
                url = 'https://www.strava.com/oauth/token',
                data = {
                    'client_id': secret_dict['client_id'],
                    'client_secret': secret_dict['client_secret'],
                    'grant_type': 'refresh_token',
                    'refresh_token': self.strava_tokens['refresh_token']
                }
            )
            # Replace the old strava_tokens with the response.
            self.strava_tokens = response.json()
            # Save new tokens to file
            with open('strava_tokens.json', 'w') as outfile:
                json.dump(self.strava_tokens, outfile)
        else:
            print("No need to update tokens.")

        # Now that that's done, go and fetch all of your activities from Strava.
        # If you've done this before, get_all_activities will just update the relevant CSV
        # file with new activities. It doesn't download everything again.
        self.get_all_activities()
        # And now separate out the different activity types and get summary info for each.
        self.separate_activity_types()




    def get_strava_profile(self, create_csv = True, profile_csv_path =
                     'profile.csv'):
        """Returns a pandas DataFrame that contains your profile information. It also prints it to a csv file unless create_csv = False."""
        # Get profile info. You need the URL, access_token and header information.
        access_token = self.strava_tokens['access_token']
        url = 'https://www.strava.com/api/v3/athlete'
        hdrs = {'Authorization': 'Bearer ' + access_token}

        # Make a request object from the info returned.
        r = requests.get(url, headers=hdrs)
        # Create a dataframe from the json content. This is just the profile info.
        # You have to orient by index because some of the fields may be empty
        # which causes a ValueError("arrays must all be same length") otherwise.
        dataframe = pd.DataFrame.from_dict(r.json(), orient="index")
        # Transpose it if you prefer to see it in the opposite orientation.
        #dataframe = dataframe.transpose()

        # Now convert the dataframe to a csv file if desired.
        if create_csv:
            dataframe.to_csv(profile_csv_path)

        return dataframe

    def get_all_activities(self, start_date="January 1, 1970", end_date = None, csv_file="strava-activities.csv", overwrite=False):
        """Retrieves all activities from start_date through end_date and outputs them to a csv file. If no end_date is specified, it assumes from start_date to the present. If no start date is specified, it assumes from the start of time. Dates are assumed to be in the UTC timezone and can be represented in human readable formats thanks to the dateutil module. If overwrite is False and the csv_file already exists, it tries to update the csv_file by changing the start_date to right after the most recent activity already present in the file. Note that the code does not check if the start_date is earlier than the earliest date already in the file. If you want to go back further in time than you already have, you will need to make sure that overwrite is True. If the start_date is after the latest date in the file, it assumes you want to delete everything before the start_date. This is to avoid gaps between the latest date in the file and the new start_date."""

        # Parse the start date by parsing it into a Datetime object.
        # Set the timezone to UTC
        startDT = parser.parse(start_date)
        timezone = pytz.timezone("UTC")
        startDT = timezone.localize(startDT)
        # Parse the end_date, but only if one is provided.
        # Otherwise, set it to the current time.
        if end_date != None:
            endDT = parser.parse(end_date)
            endDT = timezone.localize(endDT)
        else:
            endDT = datetime.datetime.now(timezone)
            # Dummy check
            if startDT >= endDT :
                print("The start time is later than the end time. Returning an empty DataFrame.")
                return pd.DataFrame()

        # If overwrite is False, check to see if the csv file already exists.
        # If so, check the latest date in the file. If the latest date is
        # after the start date, change the start date to the latest date.
        file_exists = False
        if not overwrite:
            try:
                existingDF = pd.read_csv(csv_file)
                existingDF.set_index("id", inplace=True)

                # If we made it here, clearly the file exists.
                file_exists = True

                # Get the latest date which is just the first date
                # in the dataframe.
                latest = existingDF["start_date"].iloc[0]
                latestDT = parser.parse(latest)
                # The earliest date is the last date in the dataframe.
                earliestDT = parser.parse(existingDF["start_date"].iloc[-1])
                print("Earliest date already in the CSV file is ", earliestDT)
                print("Latest date already in the CSV file is ", latestDT)
                print("start date = ", startDT)
                print("end date = ", endDT)
                if endDT < latestDT :
                    # Assume we are looking for a subset of data that we already have
                    # and just start over.
                    print("Overriding overwrite.")
                    overwrite = True
                elif latestDT > startDT :
                    print("Changing the start date to be the latest date already in the file.")
                    startDT = latestDT
                elif startDT > latestDT :
                    # Even though overwrite is false, in this situation
                    # we're going to overwrite anyway to avoid gaps in the data.
                    print("Need to override overwrite.")
                    overwrite = True
            except FileNotFoundError:
                # The file does not exist.
                print("The csv file does not already exist.")
        # Now that we have the start and end times, get the timestamps and convert them to strings
        # for inclusion in the URL.
        start_stamp = str(int(startDT.timestamp()))
        print("start_stamp = " + start_stamp)

        end_stamp = str(int(endDT.timestamp()))

        print("end_stamp = " + end_stamp)

        # Now make the API request.
        page = 1
        url = "https://www.strava.com/api/v3/activities"
        access_token = self.strava_tokens['access_token']

        # Create a dataframe to hold the results of the calls.
        # Also get the time so that we can make sure we don't go over
        # the 15 minute limit. (This is unlikely to happen and probably completely
        # unnecessary.)
        resultsDF = pd.DataFrame()
        req_start_time = time.perf_counter()
        curr_time = time.perf_counter()
        elapsed_time = 0
        # Now loop until there are no more results to get.
        while True:
            print('page = ' + str(page))
            # get page of activities from Strava
            # We're going to get 200 at a time.
            payload = {'access_token': access_token, 'after': start_stamp, 'before': end_stamp, 'per_page' : '200', 'page': str(page)}
            r = requests.get(url, params = payload)
            r = r.json()
            # If no results, then exit loop
            if (not r):
                print('Breaking when page = ' + str(page))
                break

            # Otherwise create a dataframe from the json. This holds
            # only the results from this particular page of results.
            actsDF = pd.DataFrame.from_dict(r, orient='columns')
            actsDF.set_index("id", inplace=True)
            # Now put all of those into the main storage DF.
            resultsDF = resultsDF.append(actsDF)

            # increment page.
            # Also check the time and number of calls.
            # Again this is almost certainly unnecessary.
            page += 1
            curr_time = time.perf_counter()
            elapsed_time = curr_time - req_start_time
            if (page == 101) and (elapsed_time/60.0 < 15) :
                print("Reached rate limit. Wait 15 minutes and then restart.")

        print("elapsed time =", elapsed_time, "seconds or ", elapsed_time/60, "minutes.")

        # At this point we've broken out of the loop and resultsDF holds all of our activities.
        # Do some processing and print to a csv file if we actually have results.
        if not resultsDF.empty :
            # Customize the following lines to produce the output you are interested in.
            # What you see below are the metrics that I like. You may want different ones.
            # BUT if you want to use the predict_avg_speed() method for bike rides as
            # I've written it, you need to keep "distance(miles)", "distance(km)",
            # "Feet per mile", "Meters per km" which make it possible to represent
            # the average elevation gain per unit of distance in Imperial and metric
            # of an activity. You also need to keep "average_speed(mph)" and "average_speed(kph)".
            # In a nut shell, the lines below adds columns to resultsDF.

            resultsDF["elapsed time(min)"] = resultsDF["elapsed_time"] / 60
            # Convert Elevation info from meters to feet and distance to miles
            # and add columns for them.
            resultsDF["elev_high(ft)"] = resultsDF['elev_high'] * 3.28084
            resultsDF["elev_low(ft)"] =resultsDF['elev_low'] * 3.28084
            resultsDF["elevation_gain(ft)"] = resultsDF['total_elevation_gain'] * 3.28084
            # Keep the next 6 columns if you want to use predict_avg_speed().
            resultsDF["distance(miles)"] = resultsDF['distance'] * 0.000621371
            resultsDF["distance(km)"] = resultsDF['distance']/1000
            resultsDF["Feet per mile"] = resultsDF["elevation_gain(ft)"] / resultsDF["distance(miles)"]
            resultsDF["Meters per km"] = resultsDF["total_elevation_gain"] / resultsDF["distance(km)"]
            # Convert meters per second to mph and to kph.
            resultsDF["average_speed(mph)"] = resultsDF['average_speed'] * 2.23694
            resultsDF["average_speed(kph)"] = resultsDF['average_speed'] * 3.6

            resultsDF["max_speed(mph)"] = resultsDF['max_speed'] * 2.23694
            resultsDF["max_speed(kph)"] = resultsDF['max_speed'] * 3.6
            # Write the dataframe to a csv and return the dataframe.
            if overwrite or (not file_exists) :
                print("Writing to", csv_file)
                resultsDF.to_csv(csv_file)
            else:
                print("Appending to", csv_file)
                resultsDF = resultsDF.append(existingDF)
                resultsDF.to_csv(csv_file)
        else:
            print("There were no new results. Not writing CSV.")

        return resultsDF


    def separate_activity_types(self):
        """Reads in a CSV file containing dowloaded Strava activities and creates a number of CSV files. It creates a CSV file for each activity type, a file with summary statistics for all activities, and a file for each activity type that shows a year by year summary for that type."""
        # Read in the CSV file and make a DataFrame.
        all_actsDF = pd.read_csv('strava-activities.csv', index_col="id", parse_dates=["start_date", "start_date_local"])
        # We need to make sure that all_actsDF has all of the columns that are referenced
        # in the loop below. Otherwise, the code might throw a key error. For example, if someone
        # has no heart rate data at all, stava-activities.csv won't have a max_heartrate column.
        necessary_columns = ["distance", "total_elevation_gain", "elapsed_time", "moving_time", "max_speed(mph)", "max_speed(kph)", "start_date", "elevation_gain(ft)", "max_heartrate"]
        for col in necessary_columns :
            if not col in all_actsDF.columns :
                all_actsDF[col] = np.nan
        # Get the list of unique activity types (Ride, Hike, Kayak, etc.)
        act_types = all_actsDF["type"].unique()
        # Get the list of unique years in the data.
        # Extract each year out of the data and sort them.
        years = pd.Series(d.year for d in all_actsDF["start_date"]).unique()
        years.sort()

        # Create a dataframe that will hold summary statistics for each activity.
        # The index or the set of rows is the activity types. The columns are the stats
        # we are interested in.
        stats = ["Total Distance (miles)", "Total Distance (km)", "Total Elev. Gain (meters)", "Total Elev. Gain (ft)", "Total Elev. Gain (miles)", "Total Elev. Gain (km)", "Total Duration (hours)", "Total Duration (days)", "Average Duration (min)", "Total Moving Time (hours)", "Total Moving Time (days)", "Average Moving Time (min)", "Average Speed (mph)", "Average Speed (kph)", "Max Speed (mph)", "Max Speed (kph)", "Max Speed Date", "Max Elevation Gain(ft)", "Max Elevation Gain(m)" "Max Elevation Gain Date", "Max Heart Rate", "Max HR Date"]
        summaryDF = pd.DataFrame(index=act_types, columns=stats)
        # Loop through all of the activity types and add info into the summary file.
        # Also create a csv for each activity that has the Strava info for that activity only.
        for act in act_types:
            actDF = all_actsDF[all_actsDF["type"] == act]
            actDF.to_csv(act + ".csv")
            # Add the summary stats
            summaryDF.loc[act, "Total Distance (miles)"] = actDF["distance"].sum() * 0.000621371
            summaryDF.loc[act, "Total Distance (km)"] = actDF["distance"].sum() / 1000
            summaryDF.loc[act, "Total Elev. Gain (meters)"] = actDF["total_elevation_gain"].sum()
            summaryDF.loc[act, "Total Elev. Gain (ft)"] = actDF["total_elevation_gain"].sum() * 3.28084
            summaryDF.loc[act, "Total Elev. Gain (miles)"] = actDF["total_elevation_gain"].sum() * 3.28084/5280
            summaryDF.loc[act, "Total Elev. Gain (km)"] = actDF["total_elevation_gain"].sum() / 1000
            summaryDF.loc[act, "Total Duration (hours)"] = actDF["elapsed_time"].sum() / 3600
            summaryDF.loc[act, "Total Duration (days)"] = actDF["elapsed_time"].sum() / (3600*24)
            summaryDF.loc[act, "Average Duration (min)"] = actDF["elapsed_time"].mean() / 60
            summaryDF.loc[act, "Total Moving Time (hours)"] = actDF["moving_time"].sum() / 3600
            summaryDF.loc[act, "Total Moving Time (days)"] = actDF["moving_time"].sum() / (3600*24)
            summaryDF.loc[act, "Average Moving Time (min)"] = actDF["moving_time"].mean() / 60
            summaryDF.loc[act, "Average Speed (mph)"] = (actDF["distance"].sum() / actDF["moving_time"].sum()) * 2.23694
            summaryDF.loc[act, "Average Speed (kph)"] = (actDF["distance"].sum() / actDF["moving_time"].sum()) * 3.6
            summaryDF.loc[act, "Max Speed (mph)"] = actDF["max_speed(mph)"].max()
            summaryDF.loc[act, "Max Speed (kph)"] = actDF["max_speed(kph)"].max()
            # We have to be careful anytime we want a specific date that something occured because
            # it may never have occurred and the result may be empty. That's why we do the following
            # five lines.
            s = actDF.loc[actDF["max_speed(mph)"] == actDF["max_speed(mph)"].max(), "start_date"]
            if not s.empty :
                summaryDF.loc[act, "Max Speed Date"] = s.iloc[0].date()
            else :
                summaryDF.loc[act, "Max Speed Date"] = None
            summaryDF.loc[act, "Max Elevation Gain(ft)"] = actDF["elevation_gain(ft)"].max()
            summaryDF.loc[act, "Max Elevation Gain(m)"] = actDF["total_elevation_gain"].max()
            s = actDF.loc[actDF["elevation_gain(ft)"] == actDF["elevation_gain(ft)"].max(), "start_date"]
            if not s.empty :
                summaryDF.loc[act, "Max Elevation Gain Date"] = s.iloc[0].date()
            else :
                summaryDF.loc[act, "Max Elevation Gain Date"] = None
            summaryDF.loc[act, "Max Heart Rate"] = actDF["max_heartrate"].max()
            # We have to be careful with max heart rate because not all activities will have HR data.
            # The following code makes sure there is HR data before trying to access it.
            s = actDF.loc[actDF["max_heartrate"] == actDF["max_heartrate"].max(), "start_date"]
            if not s.empty :
                summaryDF.loc[act, "Max HR Date"] = s.iloc[0].date()
            else:
                summaryDF.loc[act, "Max HR Date"] = None

            # Summarize each activity by year
            act_summaryDF = pd.DataFrame(index=stats, columns = years)
            for y in years :
                subDF = actDF[(actDF["start_date"] >= datetime.datetime(year = y, month = 1, day = 1, tzinfo=pytz.utc)) & (actDF["start_date"] < datetime.datetime(year = y+1, month = 1, day = 1, tzinfo=pytz.utc))]
                # Need to check that we had any of this activity in the year.
                if not subDF.empty :
                    act_summaryDF.loc["Total Distance (miles)", y] = subDF["distance"].sum() * 0.000621371
                    act_summaryDF.loc["Total Distance (km)", y] = subDF["distance"].sum() / 1000
                    act_summaryDF.loc["Total Elev. Gain (meters)", y] = subDF["total_elevation_gain"].sum()
                    act_summaryDF.loc["Total Elev. Gain (ft)", y] = subDF["total_elevation_gain"].sum() * 3.28084
                    act_summaryDF.loc["Total Elev. Gain (miles)", y] = subDF["total_elevation_gain"].sum() * 3.28084/5280
                    act_summaryDF.loc["Total Elev. Gain (km)", y] = subDF["total_elevation_gain"].sum() / 1000
                    act_summaryDF.loc["Total Duration (hours)", y] = subDF["elapsed_time"].sum() / 3600
                    act_summaryDF.loc["Total Duration (days)", y] = subDF["elapsed_time"].sum() / (3600*24)
                    act_summaryDF.loc["Average Duration (min)", y] = subDF["elapsed_time"].mean() / 60
                    act_summaryDF.loc["Total Moving Time (hours)", y] = subDF["moving_time"].sum() / 3600
                    act_summaryDF.loc["Total Moving Time (days)", y] = subDF["moving_time"].sum() / (3600*24)
                    act_summaryDF.loc["Average Moving Time (min)", y] = subDF["moving_time"].mean() / 60
                    act_summaryDF.loc["Average Speed (mph)", y] = (subDF["distance"].sum() / subDF["moving_time"].sum()) * 2.23694
                    act_summaryDF.loc["Average Speed (kph)", y] = (subDF["distance"].sum() / subDF["moving_time"].sum()) * 3.6
                    act_summaryDF.loc["Max Speed (mph)", y] = subDF["max_speed(mph)"].max()
                    act_summaryDF.loc["Max Speed (kph)", y] = subDF["max_speed(kph)"].max()
                    s = subDF.loc[subDF["max_speed(mph)"] == subDF["max_speed(mph)"].max(), "start_date"]
                    if not s.empty:
                        act_summaryDF.loc["Max Speed Date", y] = s.iloc[0].date()
                    else :
                        act_summaryDF.loc["Max Speed Date", y] = None

                    act_summaryDF.loc["Max Elevation Gain(ft)", y] = subDF["elevation_gain(ft)"].max()
                    act_summaryDF.loc["Max Elevation Gain(ft)", y] = subDF["total_elevation_gain"].max()
                    s = subDF.loc[subDF["elevation_gain(ft)"] == subDF["elevation_gain(ft)"].max(), "start_date"]
                    if not s.empty :
                        act_summaryDF.loc["Max Elevation Gain Date", y] = s.iloc[0].date()
                    else :
                        act_summaryDF.loc["Max Elevation Gain Date", y] = None
                    act_summaryDF.loc["Max Heart Rate", y] = subDF["max_heartrate"].max()
                    s = subDF.loc[subDF["max_heartrate"] == subDF["max_heartrate"].max(), "start_date"]
                    if not s.empty :
                        act_summaryDF.loc["Max HR Date", y] = s.iloc[0].date()
                    else:
                        act_summaryDF.loc["Max HR Date", y] = None

            # Add a few totals
            act_summaryDF.loc["Total Distance (miles)", "Total"] = act_summaryDF.loc["Total Distance (miles)"].sum()
            act_summaryDF.loc["Total Distance (km)", "Total"] = act_summaryDF.loc["Total Distance (km)"].sum()
            act_summaryDF.loc["Total Elev. Gain (meters)", "Total"] = act_summaryDF.loc["Total Elev. Gain (meters)"].sum()
            act_summaryDF.loc["Total Elev. Gain (ft)", "Total"] = act_summaryDF.loc["Total Elev. Gain (ft)"].sum()
            act_summaryDF.loc["Total Elev. Gain (miles)", "Total"] = act_summaryDF.loc["Total Elev. Gain (miles)"].sum()
            act_summaryDF.loc["Total Elev. Gain (km)", "Total"] = act_summaryDF.loc["Total Elev. Gain (km)"].sum()
            act_summaryDF.loc["Total Duration (hours)", "Total"] = act_summaryDF.loc["Total Duration (hours)"].sum()
            act_summaryDF.loc["Total Duration (days)", "Total"] = act_summaryDF.loc["Total Duration (days)"].sum()

            act_summaryDF.loc["Average Duration (min)", "Total"] = summaryDF.loc[act, "Average Duration (min)"]
            act_summaryDF.loc["Total Moving Time (hours)", "Total"] = act_summaryDF.loc["Total Moving Time (hours)"].sum()
            act_summaryDF.loc["Total Moving Time (days)", "Total"] = act_summaryDF.loc["Total Moving Time (days)"].sum()
            act_summaryDF.loc["Average Moving Time (min)", "Total"] = summaryDF.loc[act, "Average Moving Time (min)"]
            act_summaryDF.loc["Average Speed (mph)", "Total"] = summaryDF.loc[act, "Average Speed (mph)"]
            act_summaryDF.loc["Average Speed (kph)", "Total"] = summaryDF.loc[act, "Average Speed (kph)"]
            act_summaryDF.loc["Max Speed (mph)", "Total"] = act_summaryDF.loc["Max Speed (mph)"].max()
            act_summaryDF.loc["Max Speed (kph)", "Total"] = act_summaryDF.loc["Max Speed (kph)"].max()
            act_summaryDF.loc["Max Speed Date", "Total"] = summaryDF.loc[act, "Max Speed Date"]
            act_summaryDF.loc["Max Elevation Gain(ft)", "Total"] = summaryDF.loc[act, "Max Elevation Gain(ft)"]
            act_summaryDF.loc["Max Elevation Gain(m)", "Total"] = summaryDF.loc[act, "Max Elevation Gain(m)"]
            act_summaryDF.loc["Max Elevation Gain Date", "Total"] = summaryDF.loc[act, "Max Elevation Gain Date"]
            act_summaryDF.loc["Max Heart Rate", "Total"] = summaryDF.loc[act, "Max Heart Rate"]
            act_summaryDF.loc["Max HR Date", "Total"] = summaryDF.loc[act, "Max HR Date"]

            # Print the annual summary
            act_summaryDF.to_csv(act + "-by-year.csv")

        # Print the summary to a csv

        summaryDF.to_csv("strava-summary.csv")


    def predict_avg_speed(self, elev_gain, distance, lower_speed_filter=0, upper_speed_filter=float("inf"), lower_distance_filter=0, upper_distance_filter=float("inf"), model_start_year=1970, dist_fudge=0.1, elev_fudge=0.1, metric=False, activity_type="Ride") :
        """This method takes the total elevation gain of a proposed route (in feet or meters) and the distance of that route (in miles or kilometers) to predict what your average speed will be based on past performance. It does this by generating three different models (because I haven't decided on a best model yet). The first is based on a linear regression where the independent variable is average elevation gain which is just the total elevation gain of an activity (in feet or meters) divided by the distance (in miles or kilometers). The dependent variable is average speed. The second model is based on a multivariate linear regression using elevation gain and distance. The third model is just the average of the average speeds of activities with a similar distance and elevation profile.  The *_filter parameters allow you to specify activities to ignore. For example, if you don't want to include slow bike rides with your kids in your model, set the lower_speed_filter to a speed lower than your usual average speed on solo rides. model_start_year allows you to specify what data you want to include in your modeling. If you only want to include bike rides from 2020 on, for example, set model_start_year to 2020. The fudge factors let you define what it means for an activity to be similar to another in the third model. The defaults are both 0.1. This means that an activity is considered similar if it has a distance and average elevation gain within plus or minus 10% of the those of the activity you are interested in. The metric parameter lets you toggle between Imperial and metric. The default is Imperial. activity_type let's you set what type of activity you are modeling. The default is Ride. If you want to model a different activity, set the activity_type to be exactly as Strava names it: Ride, Hike, Kayaking, Canoeing, etc."""

        # Set the name of the csv file to use.
        csv_file = activity_type + ".csv"

        # First read the Ride data from the appropriate csv file and build a pandas DataFrame.
        # Put it in a try statement to catch the situation where Ride.csv does not exist.
        try :
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
            # Make the appropriate adjustments for Imperial or metric.
            if not metric :
                speed_key = "average_speed(mph)"
                dist_key = "distance(miles)"
                avg_elev_gain_key = "Feet per mile"
                elev_gain_key = "elevation_gain(ft)"
                speed_unit = "mph"
            else:
                speed_key = "average_speed(kph)"
                dist_key = "distance(km)"
                avg_elev_gain_key = "Meters per km"
                elev_gain_key = "total_elevation_gain"
                speed_unit = "kph"

            # Clean up the data to filter out irrelevant rides. I've put in a lower and upper
            # speed filters and lower and upper distance filters because I want these models
            # to ignore rides that I have taken with my family (which are slower than my solo rides)
            # and short rides in which I have not yet fully warmed up. If I want to get a sense of how
            # long a family ride will take, I use the upper filters to ignore my solo rides.
            filteredDF = actDF[ (actDF[speed_key] > lower_speed_filter) &
                                 (actDF[speed_key] < upper_speed_filter) &
                                 (actDF[dist_key] > lower_distance_filter) &
                                 (actDF[dist_key] < upper_distance_filter) &
                                 (actDF["start_date"] >= datetime.datetime(year = model_start_year, month = 1, day=1, tzinfo=pytz.utc))]
            avg_elev_gain = elev_gain/distance
            # For kicks let's plot average speed versus average elevation gain and
            # add in a line showing the linear regression.
            fig, ax = plt.subplots()
            ax.scatter(filteredDF[avg_elev_gain_key], filteredDF[speed_key])
            pred1 = 0.0
            pred2 = [0.0]
            if filteredDF[avg_elev_gain_key].sum() > 0:
                lin = linregress(filteredDF[avg_elev_gain_key], filteredDF[speed_key])
                #max_avg_elev_gain = 100
                max_avg_elev_gain = filteredDF[avg_elev_gain_key].max()
                x = np.arange(0,max_avg_elev_gain)
                y = x*lin.slope + lin.intercept
                ax.plot(x,y, color='red')
                ax.set_title("Average Speed vs. Average Elevation Gain - " + activity_type)
                if not metric:
                    ax.set_xlabel("Average elevation gain (ft) per mile")
                    ax.set_ylabel("Average Speed (mph)")
                else:
                    ax.set_xlabel("Average elevation gain (meters) per km")
                    ax.set_ylabel("Average Speed (kph)")
                if not metric :
                    fig.savefig(activity_type+"-AvgSpeedVsAvgElevGainImperial.png")
                else :
                    fig.savefig(activity_type+"-AvgSpeedVsAvgElevGainMetric.png")
                # Now do the multivariate regression.
                Indep = filteredDF[[elev_gain_key, dist_key]]
                dep = filteredDF[speed_key]
                regr = linear_model.LinearRegression()
                regr.fit(Indep,dep)
                # Now get the predictions.
                pred1 = avg_elev_gain*lin.slope + lin.intercept
                pred2 = regr.predict([[elev_gain, distance]])
                print("Average elevation gain per distance =", f'{avg_elev_gain:.2f}', avg_elev_gain_key)
                print("******** Model 1 ********")
                print("Predicted average speed based on average elevation gain per distance =", f'{pred1:.2f}', speed_unit)
                print("Moving time would equal", math.floor(distance/pred1), "hours and", round(((distance%pred1)/pred1)*60), "minutes.")
                print("******** Model 2 ********")
                print("Predicted average speed based on multivariate regression =", f'{pred2[0]:.2f}', speed_unit)
                print("Moving time would equal", math.floor(distance/pred2[0]), "hours and", round(((distance%pred2[0])/pred2[0])*60), "minutes.")
            else:
                print("Sorry. Can't use Models 1 and 2 if there is no elevation gain.")
            # To make the prediction based on similar activities we need to filter the data more
            # to include only activities that are similar to the one we're asking about in terms
            # of distance and average elevation gain. Here's where we use the fudge factors.
            similarDF = filteredDF[ (filteredDF[dist_key] >= distance - dist_fudge*distance) & (filteredDF[dist_key] <= distance + dist_fudge*distance) & (filteredDF[avg_elev_gain_key] >= avg_elev_gain - elev_fudge*avg_elev_gain) & (filteredDF[avg_elev_gain_key] <= avg_elev_gain + elev_fudge*avg_elev_gain)  ]
            similar_acts= similarDF[[speed_key, avg_elev_gain_key, dist_key]]
            pred3 = 0
            print("******** Model 3 ********")
            if len(similar_acts) > 0 :
                pred3 = similarDF[speed_key].mean()
                print("Predicted average speed based on", len(similar_acts), "similar activities =", f'{pred3:.2f}', speed_unit)
                print("Moving time would equal", math.floor(distance/pred3), "hours and", round(((distance%pred3)/pred3)*60), "minutes.")
            else :
                print("There were no similar activities. Perhaps try more generous fudge factors.")

            return [pred1, pred2[0], pred3]
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some of that activity and try again.")
