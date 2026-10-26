import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

def safe_hist(ax, series, weights, bins, color, label):
    """
    Plot a histogram only if the provided Series has at least one valid value.
    """
    s = series.dropna()
    if s.empty:
        return  # nothing to plot
    
    ax.hist(
        s,
        bins=bins,
        density=False,
        cumulative=-1,
        weights=np.ones(len(s)) * weights,
        histtype='stepfilled',
        facecolor=color,
        alpha=0.45,
        align='left',
        label=label
    )



def configure_ticks(ax, start, stop, freq_major = None, freq_minor = None):
    """Set x-axis major and minor ticks with formatting."""

    if freq_major is None:
        freq_major = '1D'  # default
    if freq_minor is None:
        freq_minor = '1h'  # default
        
    major_ticks = pd.date_range(start=start.normalize(), end=stop.normalize(), freq=freq_major)
    minor_ticks = pd.date_range(start=start, end=stop, freq= freq_minor)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.tick_params(axis='x', labelrotation=90, labelsize=8)
    plt.xticks(major_ticks, rotation=90)  # Ensure ticks are shown


def plot_temperature(ax, day, df, curves=None, colors=None, linestyles = None ):
    """
    Plot temperature lines with selectable curves.

    Parameters:
        ax : matplotlib axis
        day : array-like, x-axis values
        df  : DataFrame, containing temperature columns
        curves : list of str, column names to plot (default all available)
        colors : list of str, matplotlib colors for each curve (optional)
    """
    ax.set_ylabel('Temperature (°C)', color='k', fontsize=14)
    
    if curves is None:
        curves = ["t_i"]  # default columns
    if colors is None:
        # default matplotlib tab colors (repeated if more curves)
        default_colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple']
        colors = default_colors[:len(curves)]
    if linestyles is None:
        # default matplotlib linestyles (repeated if more curves)
        default_linestyles = ['solid', 'solid', 'solid', 'solid', 'solid']
        linestyles = default_linestyles[:len(curves)]
    
    for i, col in enumerate(curves):
        color = colors[i] if i < len(colors) else None
        linestyle = linestyles[i] if i < len(colors) else None
        ax.plot(day, df[col], color=color, linestyle =linestyle, label=col)
    
    ax.tick_params(axis='y', labelcolor='k', labelsize=14)
    ax.grid(True, which='major', linestyle='dotted', alpha=0.8)
    ax.legend()


def plot_power(ax, day, df, curves=None, colors=None, linestyles = None, rolling_span = None):
    """
    Plot power lines (rolling) with selectable curves.

    Parameters:
        ax : matplotlib axis
        day : array-like, x-axis values
        df  : DataFrame, containing temperature columns
        curves : list of str, column names to plot (default all available)
        colors : list of str, matplotlib colors for each curve (optional)
    """
    ax.set_ylabel('Power (W/m²)', color='k', fontsize=14)

    if curves is None:
        curves = ["Q_dot_g"]  # default columns
    if colors is None:
        # default matplotlib tab colors (repeated if more curves)
        default_colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:purple']
        colors = default_colors[:len(curves)]
    if linestyles is None:
        # default matplotlib linestyles (repeated if more curves)
        default_linestyles = ['solid', 'solid', 'solid', 'solid', 'solid']
        linestyles = default_linestyles[:len(curves)]
    if rolling_span is None:
        rolling_span = 4
    
    for i, col in enumerate(curves):
        color = colors[i] if i < len(colors) else None
        linestyle = linestyles[i] if i < len(colors) else None
        ax.plot(day, df[col].rolling(rolling_span, center=True, min_periods=1).mean(), color=color, linestyle =linestyle, label=col)
    
    ax.tick_params(axis='y', labelcolor='k', labelsize=14)
    ax.grid(True, which='major', linestyle='dotted', alpha=0.8)
    ax.legend()


def plot_weekly_data(dfr, title, start, stop):
    
    # ------------------------------------------------------------
    # Make start/stop timezone-naive
    # ------------------------------------------------------------
    if start.tzinfo is not None:
        start = start.tz_localize(None)

    if stop.tzinfo is not None:
        stop = stop.tz_localize(None)

    # ------------------------------------------------------------
    # Data for plotting
    # ------------------------------------------------------------
    df_plot = dfr.copy()

    if df_plot.index.tz is not None:
        df_plot.index = df_plot.index.tz_localize(None)

    # ------------------------------------------------------------
    # Select exactly one week:
    # [Monday 00:00, following Monday 00:00)
    # ------------------------------------------------------------
    mask = (df_plot.index >= start) & (df_plot.index < stop)

    # Create plot
    fig, (ax0, ax2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0},
        figsize=(12, 6)
    )

    ax0.plot(df_plot.index[mask], df_plot["t_in"][mask],
             '-', color='C0', label='t_in')
    ax0.plot(df_plot.index[mask], df_plot["t_out"][mask],
             '--', color='C7', label='t_out')
    ax0.plot(df_plot.index[mask], df_plot["t_set_h"][mask],
             ':', color='C0', label='t_set_h')

    ax2.plot(df_plot.index[mask], df_plot['Q_dot_h_Wm2'][mask],
             '-', color='C3', label='Q_dot_h_Wm2')

    # Fill between t_i and t_set where t_i < t_set
    mask_fill = (df_plot["t_in"][mask] < df_plot["t_set_h"][mask]) 
    ax0.fill_between(
        df_plot.index[mask],
        df_plot["t_in"][mask],
        df_plot["t_set_h"][mask],
        where=mask_fill,
        interpolate=True,
        alpha=0.3,
        color="orange"
    )

    # ------------------------------------------------------------
    # Major ticks: every day at 00:00
    # ------------------------------------------------------------
    major_ticks = pd.date_range(
        start=start,
        end=stop,
        freq='1D',
        inclusive='left'
    )

    # Minor ticks: every 4 hours
    minor_ticks = pd.date_range(
        start=start,
        end=stop,
        freq='4h',
        inclusive='left'
    )

    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)

    ax2.xaxis.set_major_formatter(
        mdates.DateFormatter('%d/%m/%Y')
    )
    ax2.xaxis.set_minor_formatter(
        mdates.DateFormatter('%H')
    )

    ax2.tick_params(axis='x', labelrotation=90)

    ax0.set_yticks(np.arange(-15, 31, 5))
    ax2.set_yticks(np.arange(0, 121, 20))

    # Grid
    ax0.grid(True, which='major', linestyle='solid', alpha=0.8)
    ax0.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    ax2.grid(True, which='major', linestyle='solid', alpha=0.8)
    ax2.grid(True, which='minor', linestyle='dotted', alpha=0.5)

    # Legend
    ax0.legend(ncol=15, loc='upper right')
    ax2.legend(ncol=15, loc='upper right')

    # Text
    ax0.text(
        0.07, 0.92, 'Temperature (°C)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax0.transAxes
    )

    ax2.text(
        0.09, 0.9, 'Heating Power (W/m2)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax2.transAxes
    )

    # Remove last y tick label
    yticks = ax2.get_yticklabels()
    yticks[-1].set_visible(False)

    plt.tight_layout()

    plt.suptitle(
        title + '\n' + 'Week : ' +
        start.strftime('%d/%m/%Y') + ' - ' +
        (stop - pd.Timedelta(minutes=15)).strftime('%d/%m/%Y'),
        y=1.05
    )

    plt.show()


def plot_daily_data(dfr, title, start, stop):
    
    # Convert to naive for matplotlib
    df_plot = dfr.copy()
    # df_plot.index = df_plot.index.tz_convert(None)
    
    mask = (df_plot.index >= start) & (df_plot.index <= stop)
    
    # Create plot
    fig, (ax0, ax2) = plt.subplots(nrows = 2, sharex=True, gridspec_kw={'height_ratios': [1.0, 1.0], 'hspace': 0}, figsize=(12, 7))
    
    # Plot with plot_date (forces datetime handling)
    ax0.plot(df_plot.index[mask], df_plot["t_in"][mask], '-', color='C0', label='t_in')
    ax0.plot(df_plot.index[mask], df_plot["t_out"][mask], '--', color='C7', label='t_out')
    ax0.plot(df_plot.index[mask], df_plot["t_set_h"][mask], ':', color='C0', label='t_set_h')

    ax2.plot(df_plot.index[mask], df_plot['Q_dot_h_Wm2'][mask], '-', color='C3', label='Q_dot_h_Wm2')

    # Fill between t_i and t_set where t_i < t_set
    mask_fill = (df_plot["t_in"][mask] < df_plot["t_set_h"][mask]) 
    ax0.fill_between(
        df_plot.index[mask],
        df_plot["t_in"][mask],
        df_plot["t_set_h"][mask],
        where=mask_fill,
        interpolate=True,
        alpha=0.3,
        color="orange"
    )

    # Generate major ticks (labels) every hour
    major_ticks = pd.date_range(start=start, end=stop, freq='1h')
    
    # Generate minor ticks (grid) every 15 minutes
    minor_ticks = pd.date_range(start=start, end=stop, freq='15min')
    
    ax2.set_xticks(major_ticks)  
    ax2.set_xticks(minor_ticks, minor=True)  
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y - %H:%M'))
    ax2.tick_params(axis='x', labelrotation=90)
    
    ax0.set_yticks(np.arange(-15, 31, 5))
    ax2.set_yticks(np.arange(0, 121, 20))
    
    # Grid
    ax0.grid(True, which='major', linestyle='solid', alpha=0.8)
    ax0.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    ax2.grid(True, which='major', linestyle='solid', alpha=0.8)
    ax2.grid(True, which='minor', linestyle='dotted', alpha=0.5)
    
    # Legend
    ax0.legend(ncol=15, loc='upper right')
    ax2.legend(ncol=15, loc='upper right')

    # Text
    ax0.text(0.07, 0.92, 'Temperature (°C)', horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes)
    ax2.text(0.09, 0.9, 'Heating Power (W/m2)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    
    # remove first tick label for the first subplot
    yticks = ax0.get_yticklabels()
    # yticks[0].set_visible(False)
    
    # remove first and last tick labels for the third subplot
    yticks = ax2.get_yticklabels()
    # yticks[0].set_visible(False)
    yticks[-1].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(title + '\n' + 'Day : ' + start.strftime('%d/%m/%Y') , y = 1.05)
    plt.show()       