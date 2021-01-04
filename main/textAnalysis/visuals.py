from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import main.textAnalysis.utils as utils
from datetime import datetime
from scipy.signal import savgol_filter


def wordCountDistribution(contactObj, incomingWordCounts, outgoingWordCounts):
    traces = []
    traces.append(go.Histogram(
        x=outgoingWordCounts,
        name='Outgoing Word Count Distribution'
    ))

    traces.append(go.Histogram(
        x=incomingWordCounts,
        name='Incoming Word Count Distribution'
    ))
    layout = go.Layout(

        title={
            'text': f'Word Count Distribution for {contactObj.name}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Words', color='white'),
        yaxis=dict(title='Number of Texts', color='white'),
        paper_bgcolor='rgba(0,61,81, 0.8)',
        plot_bgcolor='rgba(0,77,102, 0.8)',
        legend=dict(),
        title_font_color='white'
    )
    config = {'displaylogo': False}
    figure = go.Figure(layout=layout, data=traces)
    fig = plot(figure, output_type='div', include_plotlyjs=False, show_link=False, config=config)
    return fig


def lagDistribution(contactObj, incomingLagTimes, outgoingLagTimes):
    traces = []
    traces.append(go.Histogram(
        x=outgoingLagTimes,
        name='Outgoing Lag Time Distribution'
    ))

    traces.append(go.Histogram(
        x=incomingLagTimes,
        name='Incoming Lag Time Distribution'
    ))
    layout = go.Layout(

        title={
            'text': f'Lag Time Distribution for {contactObj.name}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Lag Time (Minutes)', color='white'),
        yaxis=dict(title='Number of Texts', color='white'),
        paper_bgcolor='rgba(0,61,81, 0.8)',
        plot_bgcolor='rgba(0,77,102, 0.8)',
        legend=dict(),
        title_font_color='white'
    )
    config = {'displaylogo': False}
    figure = go.Figure(layout=layout, data=traces)
    fig = plot(figure, output_type='div', include_plotlyjs=False, show_link=False, config=config)
    return fig


def incomingWordsOverTime(contactObj, incomingTimeStamps, incomingWordCounts):
    traces = [go.Scatter(
        x=incomingTimeStamps,
        y=incomingWordCounts,
        line_color="#ff9d00"
    )]
    layout = go.Layout(

        title={
            'text': f"Number of words in Your Texts vs. Time",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Time', color='white'),
        yaxis=dict(title='Number of Words', color='white'),
        paper_bgcolor='rgba(0,61,81, 0.8)',
        plot_bgcolor='rgba(0,77,102, 0.8)',
        legend=dict(),
        title_font_color='white'
    )

    config = {'displaylogo': False}
    figure = go.Figure(layout=layout, data=traces)
    fig = plot(figure, output_type='div', include_plotlyjs=False, show_link=False, config=config)
    return fig


def outgoingWordsOverTime(contactObj, outgoingTimeStamps, outgoingWordCounts):
    traces = [go.Scatter(
        x=outgoingTimeStamps,
        y=outgoingWordCounts,
        line_color="#2bff00"
    )]
    layout = go.Layout(

        title={
            'text': f"Number of words in {contactObj.name}'s Texts vs. Time",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Time', color='white'),
        yaxis=dict(title='Number of Words', color='white'),
        paper_bgcolor='rgba(0,61,81, 0.8)',
        plot_bgcolor='rgba(0,77,102, 0.8)',
        legend=dict(),
        title_font_color='white',
    )
    config = {'displaylogo': False}
    figure = go.Figure(layout=layout, data=traces)
    fig = plot(figure, output_type='div', include_plotlyjs=False, show_link=False, config=config)
    return fig


def visualizeContact(contactObj):
    incomingTokenized = contactObj.incoming.tokenized
    outgoingTokenized = contactObj.outgoing.tokenized
    incomingWordCounts = [len(tokens) for tokens in incomingTokenized]
    outgoingWordCounts = [len(tokens) for tokens in outgoingTokenized]
    incomingTimeStamps = contactObj.incoming.timeStamps
    outgoingTimeStamps = contactObj.outgoing.timeStamps
    '''Smoothing code taken from: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way'''
    incomingSav = savgol_filter((incomingTimeStamps, incomingWordCounts), 51, 3)
    outgoingSav = savgol_filter((outgoingTimeStamps, outgoingWordCounts), 51, 3)
    incomingTimeStampsSmooth, incomingWordCountsSmooth = incomingSav[0], incomingSav[1]
    outgoingTimeStampsSmooth, outgoingWordCountsSmooth = outgoingSav[0], outgoingSav[1]

    incomingLagTimes = contactObj.incoming.lagTimes
    outgoingLagTimes = contactObj.outgoing.lagTimes
    incomingLagTimes = [int(x/60000) for x in incomingLagTimes]
    outgoingLagTimes = [int(x/60000) for x in outgoingLagTimes]
    print(incomingLagTimes)
    print(outgoingLagTimes)
    '''time conversion code taken from: https://gist.github.com/blaylockbk/93d0946cc94d3d98d409c11cd99bf6c6'''
    incomingTimeStampsSmooth = [datetime.utcfromtimestamp(x/1000) for x in incomingTimeStampsSmooth]
    outgoingTimeStampsSmooth = [datetime.utcfromtimestamp(x/1000) for x in outgoingTimeStampsSmooth]
    figs = []
    figs.append(wordCountDistribution(contactObj, incomingWordCounts, outgoingWordCounts))
    figs.append(lagDistribution(contactObj, incomingLagTimes, outgoingLagTimes))
    figs.append(incomingWordsOverTime(contactObj, incomingTimeStampsSmooth, incomingWordCountsSmooth))
    figs.append(outgoingWordsOverTime(contactObj, outgoingTimeStampsSmooth, outgoingWordCountsSmooth))
    return figs


def visualizeAllContacts(contactsDict, nContacts=10, textCounts=True):
    if textCounts:
        allTextCounts(contactsDict, nContacts)


def allTextCounts(contactsDict, nContacts=10):
    names = utils.sortContactFrequency(contactsDict)
    incomingCounts = []
    outgoingCounts = []
    totalCounts = []
    for name in names:
        totalCounts.append(contactsDict[name].textCount)
        incomingCounts.append(len(contactsDict[name].incoming.texts))
        outgoingCounts.append(len(contactsDict[name].outgoing.texts))
    traces = []
    traces.append(go.Bar(x=names[:nContacts], y=totalCounts[:nContacts], name='Total'))
    traces.append(go.Bar(x=names[:nContacts], y=incomingCounts[:nContacts], name='Incoming'))
    traces.append(go.Bar(x=names[:nContacts], y=outgoingCounts[:nContacts], name='Outgoing'))
    # style layout
    layout = go.Layout(

        title={
            'text': f'Text Count for Top {nContacts} Contacts',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        xaxis=dict(title='Contact', color='white'),
        yaxis=dict(title='Text Count', color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(),
        title_font_color='white'
    )
    config = {'displaylogo': False}
    figure = go.Figure(layout=layout, data=traces)
    fig = plot(figure, output_type='div', include_plotlyjs=False, show_link=False, config=config)
    return fig


