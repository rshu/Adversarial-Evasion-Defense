import plotly.express as px
import pandas as pd

publications = pd.read_csv('out.csv')
data = px.data.gapminder()

data_canada = data[data.country == 'Canada']
fig = px.bar(publications, x='year', y='results',
             hover_data=['year', 'results'], color='results',
             labels={'results': 'Occurrences'}, height=400)
fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=18,
        color="black"
    )
)

fig.show()
fig.write_image("publicationCount.eps")