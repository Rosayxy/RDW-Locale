import pstats

with open("profile_results.prof", "r") as file:
    stats = pstats.Stats("profile_results.prof")
    stats.sort_stats("cumulative").print_stats(10)  # Top 10 time-consuming calls
