---
title: PlanScope London
emoji: 🏙️
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: London planning refusals by borough, plus a live classifier
---

Front end for [PlanScope London](https://github.com/samaddmirzaa/london-planscope).
The borough map is drawn from `boroughs.geojson`, built by `python -m planscope.map_data`.
The classifier calls the project's API on AWS Lambda.