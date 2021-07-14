import strava

# First create a StravaAnalyzer object
sa = strava.StravaAnalyzer()

# Uncomment the line below if you want to retrieve your Strava profile data.
#sa.get_strava_profile()

# Get the predicted average speed for a route with a particular
# elevation gain and distance. Also include a lower bound on speed,
# a lower bound on distance, and a year from which to start basing your model.
# It will use data from the start year to the present. If you want all of your data included,
# omit everything except elevation gain and distance. They are the only two required parameters.

#sa.predict_avg_speed(2000, 30, lower_speed_filter=14, lower_distance_filter=20, model_start_year=2020, dist_fudge=0.1, elev_fudge=0.1)

# This is the same as the above just with the metric flag set to true.
#sa.predict_avg_speed(609.6, 48.2803, lower_speed_filter=22.5308, lower_distance_filter=32.1869, model_start_year=2020, dist_fudge=0.1, elev_fudge=0.1, metric=True)
