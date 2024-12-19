import dash
from dash import dcc, html
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
from textblob import TextBlob

# Initialize the Dash app
app = dash.Dash(__name__)

# Load Dataset
dataset = pd.read_csv('file.csv')
dataset['labels'] = dataset['labels'].map({'good': 'positive', 'bad': 'negative', 'neutral': 'neutral'})
dataset['tweets'] = dataset['tweets'].fillna('').astype(str)

# Polarity and Subjectivity
dataset['polarity'] = dataset['tweets'].apply(lambda x: TextBlob(x).sentiment.polarity)
dataset['subjectivity'] = dataset['tweets'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Function to Generate Word Cloud Data
def generate_wordcloud_data(tweets):
    text = " ".join(tweets)
    wordcloud = WordCloud(max_words=50, background_color='white').generate(text)
    return wordcloud.words_

# Scale Word Sizes
def scale_word_sizes(word_freq, min_size=20, max_size=150):
    scaled_sizes = np.interp(list(word_freq.values()), (min(word_freq.values()), max(word_freq.values())), (min_size, max_size))
    return scaled_sizes

# Word Clouds
positive_wc = generate_wordcloud_data(dataset[dataset['labels'] == 'positive']['tweets'])
negative_wc = generate_wordcloud_data(dataset[dataset['labels'] == 'negative']['tweets'])
neutral_wc = generate_wordcloud_data(dataset[dataset['labels'] == 'neutral']['tweets'])

positive_sizes = scale_word_sizes(positive_wc)
negative_sizes = scale_word_sizes(negative_wc)
neutral_sizes = scale_word_sizes(neutral_wc)

# Sentiment Distribution
sentiment_counts = dataset['labels'].value_counts()
sentiment_bar_chart = go.Figure(go.Bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    marker=dict(color=['#8B0000', '#006400', '#FF8C00'])
))
sentiment_bar_chart.update_layout(title_text="Sentiment Distribution", template="plotly_white", height=700)

# Polarity Scatter Plots
fig_positive = go.Figure(go.Scatter(
    x=dataset[dataset['labels'] == 'positive']['polarity'],
    y=dataset[dataset['labels'] == 'positive']['subjectivity'],
    mode='markers',
    marker=dict(size=5, color='#006400'),
    text=dataset[dataset['labels'] == 'positive']['tweets']
))
fig_positive.update_layout(title="Polarity vs Subjectivity (Positive)", xaxis_title="Polarity", yaxis_title="Subjectivity")

fig_negative = go.Figure(go.Scatter(
    x=dataset[dataset['labels'] == 'negative']['polarity'],
    y=dataset[dataset['labels'] == 'negative']['subjectivity'],
    mode='markers',
    marker=dict(size=5, color='#8B0000'),
    text=dataset[dataset['labels'] == 'negative']['tweets']
))
fig_negative.update_layout(title="Polarity vs Subjectivity (Negative)", xaxis_title="Polarity", yaxis_title="Subjectivity")

fig_neutral = go.Figure(go.Scatter(
    x=dataset[dataset['labels'] == 'neutral']['polarity'],
    y=dataset[dataset['labels'] == 'neutral']['subjectivity'],
    mode='markers',
    marker=dict(size=5, color='#FF8C00'),
    text=dataset[dataset['labels'] == 'neutral']['tweets']
))
fig_neutral.update_layout(title="Polarity vs Subjectivity (Neutral)", xaxis_title="Polarity", yaxis_title="Subjectivity")

# Generate Interactive Word Networks
def generate_text_network(tweets, top_n=30):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(tweets)
    words = vectorizer.get_feature_names_out()
    co_occurrence = (X.T @ X).toarray()
    elements = []
    for i, word in enumerate(words):
        elements.append({'data': {'id': word, 'label': word}})
        for j, other_word in enumerate(words):
            if i != j and co_occurrence[i, j] > 0:
                elements.append({'data': {'source': word, 'target': other_word}})
    return elements

positive_network = generate_text_network(dataset[dataset['labels'] == 'positive']['tweets'])
negative_network = generate_text_network(dataset[dataset['labels'] == 'negative']['tweets'])
neutral_network = generate_text_network(dataset[dataset['labels'] == 'neutral']['tweets'])

# Static Word Networks (Images in assets folder)
static_positive_network = "/assets/positive_text_network.png"
static_negative_network = "/assets/negative_text_network.png"
static_neutral_network = "/assets/neutral_text_network.png"

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sentiment Analysis Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Sentiment Distribution', children=[
            dcc.Graph(figure=sentiment_bar_chart)
        ]),
             dcc.Tab(label='Polarity vs Subjectivity', children=[
            html.H3("Positive", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_positive),
            html.H3("Negative", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_negative),
            html.H3("Neutral", style={'textAlign': 'center'}),
            dcc.Graph(figure=fig_neutral)
        
        ]),
        dcc.Tab(label='Word Clouds', children=[
            html.Div([
                html.H3("Positive Word Cloud"),
                dcc.Graph(figure=go.Figure(go.Scatter(
                    x=np.random.uniform(0, 100, len(positive_wc)),
                    y=np.random.uniform(0, 100, len(positive_wc)),
                    text=list(positive_wc.keys()),
                    mode='text',
                    textfont=dict(size=positive_sizes, color='#006400')
                ))),
                html.H3("Negative Word Cloud"),
                dcc.Graph(figure=go.Figure(go.Scatter(
                    x=np.random.uniform(0, 100, len(negative_wc)),
                    y=np.random.uniform(0, 100, len(negative_wc)),
                    text=list(negative_wc.keys()),
                    mode='text',
                    textfont=dict(size=negative_sizes, color='#8B0000')
                ))),
                html.H3("Neutral Word Cloud"),
                dcc.Graph(figure=go.Figure(go.Scatter(
                    x=np.random.uniform(0, 100, len(neutral_wc)),
                    y=np.random.uniform(0, 100, len(neutral_wc)),
                    text=list(neutral_wc.keys()),
                    mode='text',
                    textfont=dict(size=neutral_sizes, color='#FF8C00')
                )))
            ])
        ]),
        dcc.Tab(label='Interactive Text Networks', children=[
            html.H3("Positive Sentiment Network"),
            cyto.Cytoscape(
                id='positive-network',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '700px'},
                elements=positive_network,
                stylesheet=[{'selector': 'node', 'style': {'background-color': '#006400', 'label': 'data(label)'}}]
            ),
            html.H3("Negative Sentiment Network"),
            cyto.Cytoscape(
                id='negative-network',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '700px'},
                elements=negative_network,
                stylesheet=[{'selector': 'node', 'style': {'background-color': '#8B0000', 'label': 'data(label)'}}]
            ),
            html.H3("Neutral Sentiment Network"),
            cyto.Cytoscape(
                id='neutral-network',
                layout={'name': 'circle'},
                style={'width': '100%', 'height': '700px'},
                elements=neutral_network,
                stylesheet=[{'selector': 'node', 'style': {'background-color': '#FF8C00', 'label': 'data(label)'}}]
            )
        ]),
        dcc.Tab(label='Static Word Networks', children=[
            html.H3("Static Positive Word Network"),
            html.Img(src=static_positive_network, style={'width': '100%'}),
            html.H3("Static Negative Word Network"),
            html.Img(src=static_negative_network, style={'width': '100%'}),
            html.H3("Static Neutral Word Network"),
            html.Img(src=static_neutral_network, style={'width': '100%'})
        ])
    ])
])

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))  # Render provides the PORT via an environment variable
    app.run_server(debug=False, host="0.0.0.0", port=port)