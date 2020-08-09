import plotly.express as px
import pandas as pd

df = pd.read_csv('type.csv')

fig = px.bar(df, y='Count', x='Tasks', text='Count')
fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis_tickangle=20, bargap=0.5)

fig.update_layout(
    font=dict(
        family="Times New Roman",
        size=18,
        color="black"
    )
)
fig.show()